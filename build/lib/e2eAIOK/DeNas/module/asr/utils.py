import torch

from asr.supernet_asr import TransformerASRSuper


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