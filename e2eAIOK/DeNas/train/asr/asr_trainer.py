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
from trainer.TorchTrainer import BaseTrainer
from trainer.model.asr.asr_model_builder import ASRModelBuilder
from trainer.model.asr.init_asr_parser import parse_args
import extend_distributed as ext_dist

class ASRTrainer(BaseTrainer):
    def __init__(self, args):
        self.args = parse_args(args)
        with open(self.args.param_file, 'r') as f:
            self.hparams = yaml.safe_load(f)
        print(F"ASR args:{self.args}")
        torch.manual_seed(self.args.seed)
        if self.args.distributed_launch or (int(os.environ.get('WORLD_SIZE', 1)) > 1):
            ext_dist.init_distributed(backend=self.args.distributed_backend)
            world_size = dist.get_world_size()
        else:
            world_size = 1
        init_log(self.args.distributed_launch)
        self.model_builder = ASRModelBuilder(self.args)

    def train_one_epoch(self, model, optimizer, train_set, epoch, hparams, scheduler, feat_proc):
        logger = logging.getLogger("train")

        augment = SpecAugment(**hparams["augmentation"])
        if train_set.sampler is not None and hasattr(train_set.sampler, "set_epoch"):
            train_set.sampler.set_epoch(epoch)
        model.train()

        step = 0
        nonfinite_count = 0
        total_step = len(train_set)
        avg_train_loss = 0.0
        epoch_start_time = time.time()
        for batch in train_set:
            step += 1
            step_start_time = time.time()
            should_step = step % hparams["grad_accumulation_factor"] == 0
            wavs, wav_lens = batch.sig
            tokens_bos, _ = batch.tokens_bos
            feats = feat_proc(wavs)
            feats = model["normalize"](feats, wav_lens, epoch=epoch)
            feats = augment(feats)

            src = model["CNN"](feats)
            enc_out, pred = model["Transformer"](src, tokens_bos, wav_lens, pad_idx=hparams["pad_index"])

            logits = model["ctc_lin"](enc_out)
            p_ctc = logits.log_softmax(dim=-1)

            pred = model["seq_lin"](pred)
            p_seq = pred.log_softmax(dim=-1)

            ids = batch.id
            tokens_eos, tokens_eos_lens = batch.tokens_eos
            tokens, tokens_lens = batch.tokens

            loss_seq = kldiv_loss(p_seq, tokens_eos, length=tokens_eos_lens, label_smoothing=hparams["label_smoothing"], reduction=hparams["loss_reduction"]).sum()
            loss_ctc = ctc_loss(p_ctc, tokens, wav_lens, tokens_lens, blank_index=hparams["blank_index"], reduction=hparams["loss_reduction"]).sum()

            loss = (hparams["ctc_weight"] * loss_ctc + (1 - hparams["ctc_weight"]) * loss_seq)
            (loss / hparams["grad_accumulation_factor"]).backward()

            if should_step:
                is_loss_finite, nonfinite_count = check_gradients(model, loss, hparams["max_grad_norm"], nonfinite_count)
                if is_loss_finite:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler(optimizer)

            train_loss = loss.detach().cpu()
            avg_train_loss = update_average(train_loss, avg_train_loss, step)
            logger.info(f"epoch: {epoch}, step: {step}|{total_step}, time: {(time.time()-step_start_time):.2f}s, loss: {train_loss}, avg_loss: {avg_train_loss:.4f}, lr: {scheduler.current_lr}")

        logger.info(f"epoch: {epoch}, time: {(time.time()-epoch_start_time):.2f}s, avg_loss: {avg_train_loss:.4f}")

    def evaluate(self, model, valid_set, epoch, hparams, tokenizer, feat_proc):
        logger = logging.getLogger("evaluate")
        acc_metric = AccuracyStats()
        wer_metric = ErrorRateStats()
        model.eval()
        avg_valid_loss = 0.0
        total_step = len(valid_set)
        step = 0
        eval_start_time = time.time()
        with torch.no_grad():
            for batch in valid_set:
                step += 1
                step_start_time = time.time()
                wavs, wav_lens = batch.sig
                tokens_bos, _ = batch.tokens_bos
                feats = feat_proc(wavs)
                feats = model["normalize"](feats, wav_lens, epoch=epoch)

                src = model["CNN"](feats)
                enc_out, pred = model["Transformer"](src, tokens_bos, wav_lens, pad_idx=hparams["pad_index"])

                logits = model["ctc_lin"](enc_out)
                p_ctc = logits.log_softmax(dim=-1)

                pred = model["seq_lin"](pred)
                p_seq = pred.log_softmax(dim=-1)

                # hyps, _ = searcher(enc_out.detach(), wav_lens)

                ids = batch.id
                tokens_eos, tokens_eos_lens = batch.tokens_eos
                tokens, tokens_lens = batch.tokens

                loss_seq = kldiv_loss(p_seq, tokens_eos, length=tokens_eos_lens, label_smoothing=hparams["label_smoothing"], reduction=hparams["loss_reduction"]).sum()
                loss_ctc = ctc_loss(p_ctc, tokens, wav_lens, tokens_lens, blank_index=hparams["blank_index"], reduction=hparams["loss_reduction"]).sum()

                loss = (hparams["ctc_weight"] * loss_ctc + (1 - hparams["ctc_weight"]) * loss_seq)
                predicted_words = [tokenizer.decode_ids(utt_seq.tolist()).split(" ") for utt_seq in p_seq.argmax(-1)]
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

    def fit(self):
        logger = logging.getLogger("train")
        hparams = self.hparams
        create_experiment_directory(self.args.output_folder)
        tokenizer = sp.SentencePieceProcessor()
        train_data, valid_data, test_datasets = dataio_prepare(self.args, hparams, tokenizer)
        # load tokenizer
        load_spm(tokenizer, self.args.tokenizer_ckpt)
        model = self.model_builder.create_model(self.hparams)
        train_dataloader_opts = hparams["train_dataloader_opts"]
        valid_dataloader_opts = hparams["valid_dataloader_opts"]
        # train_dataloader = make_dataloader(train_data, 'train', self.args.distributed_launch, **hparams["train_dataloader_opts"])
        # valid_dataloader = make_dataloader(valid_data, 'valid', self.args.distributed_launch, **hparams["valid_dataloader_opts"])
        train_dataloader = get_dataloader(train_data, hparams["train_dataloader_opts"]["batch_size"], self.args.distributed_launch)
        valid_dataloader = get_dataloader(valid_data, hparams["valid_dataloader_opts"]["batch_size"], False)
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr_adam"], betas=(0.9, 0.98), eps=0.000000001)

        scheduler = NoamScheduler(lr_initial=hparams["lr_adam"], n_warmup_steps=hparams["n_warmup_steps"])
        feat_proc = Fbank(**hparams["compute_features"])
        train_start_time = time.time()

        for epoch in range(1, hparams["epochs"]+1):
            self.train_one_epoch(model, optimizer, train_dataloader, epoch, hparams, scheduler, feat_proc)
            wer = self.evaluate(model, valid_dataloader, epoch, hparams, tokenizer, feat_proc)
            if wer <= hparams["metric_threshold"]:
                logger.info(f"wer {wer} got threshold {hparams['metric_threshold']}, early stop")
                break
        logger.info(f"training time: {time.time() - train_start_time}")
