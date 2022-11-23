import os
import sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
cls_path = current_dir.rsplit("/", 3)[0]
sys.path.append(cls_path)
import sys
import torch
import logging
import time
import yaml
import sentencepiece as sp
import torch.distributed as dist

from e2eAIOK.DeNas.asr.data.dataio.dataloader import get_dataloader, make_dataloader
from e2eAIOK.DeNas.asr.data.dataio.dataset import dataio_prepare
from e2eAIOK.DeNas.asr.utils.utils import check_gradients, update_average, create_experiment_directory, init_log
from e2eAIOK.DeNas.asr.utils.parameter_transfer import load_torch_model, load_spm
from e2eAIOK.DeNas.asr.trainer.losses import ctc_loss, kldiv_loss
from e2eAIOK.DeNas.asr.trainer.schedulers import NoamScheduler
from e2eAIOK.DeNas.asr.data.augment import SpecAugment
from e2eAIOK.DeNas.asr.data.features import Fbank
from e2eAIOK.DeNas.asr.utils.Accuracy import AccuracyStats
from e2eAIOK.DeNas.asr.utils.metric_stats import ErrorRateStats
from e2eAIOK.common.trainer.torch_trainer import TorchTrainer

class ASRTrainer(TorchTrainer):
    def __init__(self, cfg):
        super(ASRTrainer, self).__init__(cfg)
        torch.manual_seed(self.cfg.seed)
        init_log(self.size > 1)
        self.tokenizer = sp.SentencePieceProcessor()
        logger = logging.getLogger("Trainer")
        logger.info(f"ASR config: {self.cfg}")
    
    def create_dataloader(self):
        self.tokenizer = sp.SentencePieceProcessor()
        train_data, valid_data, test_datasets = dataio_prepare(self.cfg, self.tokenizer)
        # load tokenizer
        load_spm(self.tokenizer, self.cfg.tokenizer_ckpt)
        train_dataloader = get_dataloader(train_data, self.cfg["train_dataloader_opts"]["batch_size"], self.size >1)
        valid_dataloader = get_dataloader(valid_data, self.cfg["valid_dataloader_opts"]["batch_size"], False)
        self.data_loader = {'train':train_dataloader, 'val':valid_dataloader}
    
    def preparation(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg["lr_adam"], betas=(0.9, 0.98), eps=0.000000001)
        self.scheduler = NoamScheduler(lr_initial=self.cfg["lr_adam"], n_warmup_steps=self.cfg["n_warmup_steps"])
        self.feat_proc = Fbank(**self.cfg["compute_features"])
        self.all_operations = {'optimizer':self.optimizer,'lr_scheduler':self.scheduler, 'feat_proc':self.feat_proc}

    def train_one_epoch(self, epoch):
        logger = logging.getLogger("train")

        augment = SpecAugment(**self.cfg["augmentation"])
        if self.data_loader['train'].sampler is not None and hasattr(self.data_loader['train'].sampler, "set_epoch"):
            self.data_loader['train'].sampler.set_epoch(epoch)
        self.model.train()

        step = 0
        nonfinite_count = 0
        total_step = len(self.data_loader['train'])
        avg_train_loss = 0.0
        epoch_start_time = time.time()
        for batch in self.data_loader['train']:
            step += 1
            step_start_time = time.time()
            should_step = step % self.cfg["grad_accumulation_factor"] == 0
            wavs, wav_lens = batch.sig
            tokens_bos, _ = batch.tokens_bos
            feats = self.feat_proc(wavs)
            feats = self.model["normalize"](feats, wav_lens, epoch=epoch)
            feats = augment(feats)

            src = self.model["CNN"](feats)
            enc_out, pred = self.model["Transformer"](src, tokens_bos, wav_lens, pad_idx=self.cfg["pad_index"])

            logits = self.model["ctc_lin"](enc_out)
            p_ctc = logits.log_softmax(dim=-1)

            pred = self.model["seq_lin"](pred)
            p_seq = pred.log_softmax(dim=-1)

            ids = batch.id
            tokens_eos, tokens_eos_lens = batch.tokens_eos
            tokens, tokens_lens = batch.tokens

            loss_seq = kldiv_loss(p_seq, tokens_eos, length=tokens_eos_lens, label_smoothing=self.cfg["label_smoothing"], reduction=self.cfg["loss_reduction"]).sum()
            loss_ctc = ctc_loss(p_ctc, tokens, wav_lens, tokens_lens, blank_index=self.cfg["blank_index"], reduction=self.cfg["loss_reduction"]).sum()

            loss = (self.cfg["ctc_weight"] * loss_ctc + (1 - self.cfg["ctc_weight"]) * loss_seq)
            (loss / self.cfg["grad_accumulation_factor"]).backward()

            if should_step:
                is_loss_finite, nonfinite_count = check_gradients(self.model, loss, self.cfg["max_grad_norm"], nonfinite_count)
                if is_loss_finite:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler(self.optimizer)

            train_loss = loss.detach().cpu()
            avg_train_loss = update_average(train_loss, avg_train_loss, step)
            logger.info(f"epoch: {epoch}, step: {step}|{total_step}, time: {(time.time()-step_start_time):.2f}s, loss: {train_loss}, avg_loss: {avg_train_loss:.4f}, lr: {self.scheduler.current_lr}")

        logger.info(f"epoch: {epoch}, time: {(time.time()-epoch_start_time):.2f}s, avg_loss: {avg_train_loss:.4f}")

    def evaluate(self, epoch):
        logger = logging.getLogger("evaluate")
        acc_metric = AccuracyStats()
        wer_metric = ErrorRateStats()
        self.model.eval()
        avg_valid_loss = 0.0
        total_step = len(self.data_loader['val'])
        step = 0
        eval_start_time = time.time()
        with torch.no_grad():
            for batch in self.data_loader['val']:
                step += 1
                step_start_time = time.time()
                wavs, wav_lens = batch.sig
                tokens_bos, _ = batch.tokens_bos
                feats = self.feat_proc(wavs)
                feats = self.model["normalize"](feats, wav_lens, epoch=epoch)

                src = self.model["CNN"](feats)
                enc_out, pred = self.model["Transformer"](src, tokens_bos, wav_lens, pad_idx=self.cfg["pad_index"])

                logits = self.model["ctc_lin"](enc_out)
                p_ctc = logits.log_softmax(dim=-1)

                pred = self.model["seq_lin"](pred)
                p_seq = pred.log_softmax(dim=-1)

                # hyps, _ = searcher(enc_out.detach(), wav_lens)

                ids = batch.id
                tokens_eos, tokens_eos_lens = batch.tokens_eos
                tokens, tokens_lens = batch.tokens

                loss_seq = kldiv_loss(p_seq, tokens_eos, length=tokens_eos_lens, label_smoothing=self.cfg["label_smoothing"], reduction=self.cfg["loss_reduction"]).sum()
                loss_ctc = ctc_loss(p_ctc, tokens, wav_lens, tokens_lens, blank_index=self.cfg["blank_index"], reduction=self.cfg["loss_reduction"]).sum()

                loss = (self.cfg["ctc_weight"] * loss_ctc + (1 - self.cfg["ctc_weight"]) * loss_seq)
                predicted_words = [self.tokenizer.decode_ids(utt_seq.tolist()).split(" ") for utt_seq in p_seq.argmax(-1)]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                wer_metric.append(ids, predicted_words, target_words)
                acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

                eval_loss = loss.detach().cpu()
                avg_valid_loss = update_average(eval_loss, avg_valid_loss, step)
                # logger.info(f"epoch: {epoch}, step: {step}|{total_step}, time: {(time.time()-step_start_time):.2f}s, loss: {eval_loss}, avg_loss: {avg_valid_loss:.4f}")

            acc = acc_metric.summarize()
            wer = wer_metric.summarize("error_rate")
            logger.info(f"epoch: {epoch}, time: {time.time()-eval_start_time}, wer: {wer}, acc: {acc}, avg_loss: {avg_valid_loss}")
        return wer