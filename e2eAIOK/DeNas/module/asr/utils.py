import torch
import os
import logging
from easydict import EasyDict as edict

from e2eAIOK.DeNas.asr.supernet_asr import TransformerASRSuper
from e2eAIOK.DeNas.asr.model_builder_denas_asr import ModelBuilderASRDeNas

logger = logging.getLogger("Utils")
supernet_config = {
    "input_shape": [8, 10, 80],
    "num_blocks": 3,
    "num_layers_per_block": 1,
    "out_channels": [64, 64, 64],
    "kernel_sizes": [5, 5, 1],
    "strides": [2, 2, 1],
    "residuals": [False, False, True],
    "input_size": 1280, #n_mels/strides * out_channels
    "d_model": 512,
    "encoder_heads": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    "nhead": 4,
    "num_encoder_layers": 12,
    "num_decoder_layers": 6,
    "mlp_ratio": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
    "d_ffn": 2048,
    "transformer_dropout": 0.1,
    "output_neurons": 5000,
    "best_model_structure": None
}

def gen_transformer(
    input_size=1280, output_neurons=5000, d_model=512, encoder_heads=[4]*12,
    nhead=4, num_encoder_layers=12, num_decoder_layers=6, mlp_ratio=[4.0]*12, 
    d_ffn=2048, transformer_dropout=0.1
):
    model = TransformerASRSuper(
        input_size = input_size,
        tgt_vocab = output_neurons,
        d_model = d_model,
        encoder_heads = encoder_heads,
        nhead = nhead,
        num_encoder_layers = num_encoder_layers,
        num_decoder_layers = num_decoder_layers,
        mlp_ratio = mlp_ratio,
        d_ffn = d_ffn,
        dropout = transformer_dropout,
        activation = torch.nn.GELU,
        normalize_before = True
    )
    return model

def load_pretrained_model(ckpt):
    if not os.path.exists(ckpt):
        raise RuntimeError(f"Can not find pre-trained model {ckpt}!")
    logger.info(f"loading pretrained model at {ckpt}")

    cfg = edict(supernet_config)
    super_model = ModelBuilderASRDeNas(cfg)._init_model()
    super_model_list = torch.nn.ModuleList([super_model["CNN"], super_model["Transformer"], super_model["seq_lin"], super_model["ctc_lin"]])
    pretrained_dict = torch.load(ckpt, map_location=torch.device('cpu'))
    super_model_list_dict = super_model_list.state_dict()
    super_model_list_keys = list(super_model_list_dict.keys())
    pretrained_keys = pretrained_dict.keys()
    for i, key in enumerate(pretrained_keys):
        super_model_list_dict[super_model_list_keys[i]].copy_(pretrained_dict[key])

    return super_model["Transformer"]