import torch
import sentencepiece as sp
from easydict import EasyDict as edict
from e2eAIOK.DeNas.asr.model_builder_denas_asr import ModelBuilderASRDeNas
from e2eAIOK.common.trainer.data.asr.data_builder_librispeech import DataBuilderLibriSpeech
from e2eAIOK.DeNas.asr.asr_trainer import ASRTrainer
from e2eAIOK.DeNas.asr.trainer.schedulers import NoamScheduler
from e2eAIOK.DeNas.asr.trainer.losses import ctc_loss, kldiv_loss
from e2eAIOK.DeNas.asr.utils.metric_stats import ErrorRateStats

class TestDeNasASRTrainer:
    '''
    Test Unified API ASRTrainer.fit()
    '''  
    def test_asr_trainer(self):
        model_structure = "(12, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 512)"
        with open("./best_model_structure.txt", 'w') as f:
            f.write(str(model_structure))
        cfg = edict({
            'domain': 'asr', 'seed': 74443, 'device': 'cpu', 'dist_backend': 'gloo',
            'best_model_structure': './best_model_structure.txt', 'data_folder': '/home/vmagent/app/dataset/LibriSpeech',
            'skip_prep': False,
            'train_csv': '/home/vmagent/app/dataset/LibriSpeech-denas/train-clean-100.csv',
            'valid_csv': '/home/vmagent/app/dataset/LibriSpeech-denas/dev-test.csv',
            'test_csv': '/home/vmagent/app/dataset/LibriSpeech-denas/dev-test.csv',
            'tokenizer_ckpt': '/home/vmagent/app/dataset/LibriSpeech/tokenizer.ckpt',
            'train_epochs': 1, 'eval_epochs': 1, 'train_batch_size': 32, 'eval_batch_size': 1,
            'num_workers': 1, 'ctc_weight': 0.3, 'grad_accumulation_factor': 1,
            'max_grad_norm': 5.0, 'loss_reduction': 'batchmean', 'sorting': 'random',
            'metric_threshold': 25, 'lr_adam': 0.001, 'sample_rate': 16000, 'n_fft': 400, 'n_mels': 80,
            'input_shape': [8, 10, 80], 'num_blocks': 3, 'num_layers_per_block': 1,
            'out_channels': [64, 64, 64], 'kernel_sizes': [5, 5, 1], 'strides': [2, 2, 1], 
            'residuals': [False, False, True], 'input_size': 1280, 'd_model': 512,
            'encoder_heads': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'nhead': 4, 'num_encoder_layers': 12,
            'num_decoder_layers': 6, 'mlp_ratio': [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            'd_ffn': 2048, 'transformer_dropout': 0.1, 'output_neurons': 5000,
            'blank_index': 0, 'label_smoothing': 0.0, 'pad_index': 0, 'bos_index': 1, 'eos_index': 2,
            'min_decode_ratio': 0.0, 'max_decode_ratio': 1.0, 'lm_weight': 0.60, 'ctc_weight_decode': 0.40,
            'n_warmup_steps': 2500, 
            'augmentation': {'time_warp': False, 'time_warp_window': 5, 'time_warp_mode': 'bicubic',
            'freq_mask': True, 'n_freq_mask': 4, 'time_mask': True, 'n_time_mask': 4,
            'replace_with_zero': False, 'freq_mask_width': 15, 'time_mask_width': 20},
            'speed_perturb': True,
            'compute_features': {'sample_rate': 16000, 'n_fft': 400, 'n_mels': 80}
        })
        flag = True
        try:
            model = ModelBuilderASRDeNas(cfg).create_model()
            tokenizer = sp.SentencePieceProcessor()
            train_dataloader, eval_dataloader = DataBuilderLibriSpeech(cfg, tokenizer).get_dataloader()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr_adam"], betas=(0.9, 0.98), eps=0.000000001)
            criterion = {"ctc_loss": ctc_loss, "seq_loss": kldiv_loss}
            scheduler = NoamScheduler(lr_initial=cfg["lr_adam"], n_warmup_steps=cfg["n_warmup_steps"])
            metric = ErrorRateStats()
            trainer = ASRTrainer(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric, tokenizer)
            trainer.fit()
        except Exception:
            flag = False

        assert flag == True