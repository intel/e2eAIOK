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

from asr.data.dataio.dataloader import get_dataloader, make_dataloader
from asr.data.dataio.dataset import dataio_prepare
from asr.utils.utils import check_gradients, update_average, create_experiment_directory, init_log
from asr.utils.parameter_transfer import load_torch_model, load_spm
from asr.trainer.losses import ctc_loss, kldiv_loss
from asr.trainer.schedulers import NoamScheduler
from asr.data.augment import SpecAugment
from asr.data.features import Fbank
from asr.utils.Accuracy import AccuracyStats
from asr.utils.metric_stats import ErrorRateStats
from e2eAIOK.common.trainer.torch_trainer import TorchTrainer
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist

class ASRTrainer(TorchTrainer):
    def __init__(self, cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric, tokenizer):
        super(ASRTrainer, self).__init__(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric)
        self.tokenizer = tokenizer
    
    def _pre_process(self):
        super()._pre_process()
        load_spm(self.tokenizer, self.cfg.tokenizer_ckpt)
        self.feat_proc = Fbank(**self.cfg["compute_features"])
    
    def _is_early_stop(self, metric):
        return metric <= self.cfg["metric_threshold"]
    
    def _dist_wrapper(self):
        if ext_dist.my_size > 1:
            for name, module in self.model.items():
                if any(p.requires_grad for p in module.parameters()):
                    module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
                    module = ext_dist.DDP(module)
                    self.model[name] = module

    def train_one_epoch(self, epoch):
        augment = SpecAugment(**self.cfg["augmentation"])
        if self.train_dataloader.sampler is not None and hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(epoch)
        self.model.train()

        step = 0
        nonfinite_count = 0
        total_step = len(self.train_dataloader)
        avg_train_loss = 0.0
        epoch_start_time = time.time()
        for batch in self.train_dataloader:
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

            # loss_seq = kldiv_loss(p_seq, tokens_eos, length=tokens_eos_lens, label_smoothing=self.cfg["label_smoothing"], reduction=self.cfg["loss_reduction"]).sum()
            # loss_ctc = ctc_loss(p_ctc, tokens, wav_lens, tokens_lens, blank_index=self.cfg["blank_index"], reduction=self.cfg["loss_reduction"]).sum()
            loss_seq = self.criterion["seq_loss"](p_seq, tokens_eos, length=tokens_eos_lens, label_smoothing=self.cfg["label_smoothing"], reduction=self.cfg["loss_reduction"]).sum()
            loss_ctc = self.criterion["ctc_loss"](p_ctc, tokens, wav_lens, tokens_lens, blank_index=self.cfg["blank_index"], reduction=self.cfg["loss_reduction"]).sum()

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
            self.logger.info(f"epoch: {epoch}, step: {step}|{total_step}, time: {(time.time()-step_start_time):.2f}s, loss: {train_loss}, avg_loss: {avg_train_loss:.4f}, lr: {self.scheduler.current_lr}")

        self.logger.info(f"epoch: {epoch}, time: {(time.time()-epoch_start_time):.2f}s, avg_loss: {avg_train_loss:.4f}")

    def evaluate(self, epoch):
        self.metric.clear()
        self.model.eval()
        avg_valid_loss = 0.0
        total_step = len(self.eval_dataloader)
        step = 0
        eval_start_time = time.time()
        with torch.no_grad():
            for batch in self.eval_dataloader:
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

                # loss_seq = kldiv_loss(p_seq, tokens_eos, length=tokens_eos_lens, label_smoothing=self.cfg["label_smoothing"], reduction=self.cfg["loss_reduction"]).sum()
                # loss_ctc = ctc_loss(p_ctc, tokens, wav_lens, tokens_lens, blank_index=self.cfg["blank_index"], reduction=self.cfg["loss_reduction"]).sum()
                loss_seq = self.criterion["seq_loss"](p_seq, tokens_eos, length=tokens_eos_lens, label_smoothing=self.cfg["label_smoothing"], reduction=self.cfg["loss_reduction"]).sum()
                loss_ctc = self.criterion["ctc_loss"](p_ctc, tokens, wav_lens, tokens_lens, blank_index=self.cfg["blank_index"], reduction=self.cfg["loss_reduction"]).sum()

                loss = (self.cfg["ctc_weight"] * loss_ctc + (1 - self.cfg["ctc_weight"]) * loss_seq)
                predicted_words = [self.tokenizer.decode_ids(utt_seq.tolist()).split(" ") for utt_seq in p_seq.argmax(-1)]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                self.metric.append(ids, predicted_words, target_words)

                eval_loss = loss.detach().cpu()
                avg_valid_loss = update_average(eval_loss, avg_valid_loss, step)

            wer = self.metric.summarize("error_rate")
            self.logger.info(f"epoch: {epoch}, time: {time.time()-eval_start_time}, wer: {wer}, avg_loss: {avg_valid_loss}")
        return wer