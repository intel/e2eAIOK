import torch
from torch import nn
from typing import Optional

import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
cls_path = current_dir.rsplit("/", 1)[0]
sys.path.append(cls_path)

from module.asr.linear import Linear
from asr.TransformerBase import (
    get_lookahead_mask,
    get_key_padding_mask,
    NormalizedEmbedding,
    PositionalEncoding
)
from module.asr.encoder import TransformerEncoder
from module.asr.decoder import TransformerDecoder
from asr.data.dataio.dataio import length_to_mask


class TransformerASRSuper(nn.Module):
    def __init__(
        self,
        tgt_vocab,
        input_size,
        d_model=512,
        encoder_heads=[4],
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        mlp_ratio=[4],
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        normalize_before=False,
        max_length: Optional[int] = 2500,
    ):
        super().__init__()
        self.encoder_kdim = None
        self.encoder_vdim = None
        self.decoder_kdim = None
        self.decoder_vdim = None

        self.positional_encoding = PositionalEncoding(d_model, max_length)

        # initialize the encoder
        self.encoder = TransformerEncoder(
            nhead=encoder_heads,
            num_layers=num_encoder_layers,
            mlp_ratio=mlp_ratio,
            d_model=d_model,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            kdim=self.encoder_kdim,
            vdim=self.encoder_vdim,
        )
        # initialize the decoder
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            d_model=d_model,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            kdim=self.decoder_kdim,
            vdim=self.decoder_vdim,
        )

        self.custom_src_module = nn.ModuleList([
            Linear(
                input_size=input_size,
                n_neurons=d_model,
                bias=True,
            ),
            torch.nn.Dropout(dropout),
        ])
        self.custom_tgt_module = nn.ModuleList([
            NormalizedEmbedding(d_model, tgt_vocab)
        ])

        # reset parameters using xavier_normal_
        self._init_params()

    def forward(self, src, tgt, wav_len=None, pad_idx=0):
        encoder_out, src_key_padding_mask = self.encode(src, wav_len)

        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_mask(tgt)

        for layer in self.custom_tgt_module:
            tgt = layer(tgt)

        tgt = tgt + self.positional_encoding(tgt)
        pos_embs_target = None
        pos_embs_encoder = None

        decoder_out, _, _ = self.decoder(
            tgt=tgt,
            memory=encoder_out,
            memory_mask=src_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )

        return encoder_out, decoder_out

    def make_masks(self, src, tgt, wav_len=None, pad_idx=0):
        src_key_padding_mask = None
        if wav_len is not None:
            abs_len = torch.round(wav_len * src.shape[1])
            src_key_padding_mask = ~length_to_mask(abs_len).bool()

        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_mask(tgt)
        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask

    @torch.no_grad()
    def decode(self, tgt, encoder_out, enc_len=None):
        tgt_mask = get_lookahead_mask(tgt)
        src_key_padding_mask = None
        if enc_len is not None:
            src_key_padding_mask = (1 - length_to_mask(enc_len)).bool()

        for layer in self.custom_tgt_module:
            tgt = layer(tgt)
        tgt = tgt + self.positional_encoding(tgt)  # add the encodings here
        pos_embs_target = None
        pos_embs_encoder = None

        prediction, self_attns, multihead_attns = self.decoder(
            tgt,
            encoder_out,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )
        return prediction, multihead_attns[-1]

    def encode(self, src, wav_len=None):
        # reshape the src vector to [Batch, Time, Fea] if a 4d vector is given
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        src_key_padding_mask = None
        if wav_len is not None:
            abs_len = torch.round(wav_len * src.shape[1])
            src_key_padding_mask = ~length_to_mask(abs_len).bool()

        for layer in self.custom_src_module:
            src = layer(src)

        src = src + self.positional_encoding(src)
        pos_embs_source = None

        encoder_out, _ = self.encoder(
            src=src,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_source,
        )
        return encoder_out, src_key_padding_mask

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

    def calc_sampled_param_num(self):
        numel = 0
        numel += self.encoder.calc_sampled_param_num()
        numel += self.custom_src_module[0].calc_sampled_param_num()
        return numel

    @classmethod
    def gen_transformer(config):
        model = TransformerASRSuper(
            input_size = config["input_size"],
            tgt_vocab = config["output_neurons"],
            d_model = config["d_model"],
            encoder_heads = config["encoder_heads"],
            nhead = config["nhead"],
            num_encoder_layers = config["num_encoder_layers"],
            num_decoder_layers = config["num_decoder_layers"],
            mlp_ratio = config["mlp_ratio"],
            d_ffn = config["d_ffn"],
            dropout = config["transformer_dropout"],
            activation = torch.nn.GELU,
            normalize_before = True
        )
        return model