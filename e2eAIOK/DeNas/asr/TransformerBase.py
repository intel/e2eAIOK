import math
import torch
import torch.nn as nn
from typing import Optional

from e2eAIOK.DeNas.module.asr.attention import RelPosEncXL, MultiheadAttention, RelPosMHAXL, PositionalwiseFeedForward
from e2eAIOK.DeNas.module.asr.normalization import LayerNorm
from e2eAIOK.DeNas.module.asr.embedding import Embedding


class TransformerBase(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        custom_src_module=None,
        custom_tgt_module=None,
        normalize_before=True,
        max_length: Optional[int] = 2500,
        encoder_kdim: Optional[int] = None,
        encoder_vdim: Optional[int] = None,
        decoder_kdim: Optional[int] = None,
        decoder_vdim: Optional[int] = None,
    ):
        super().__init__()
        self.encoder_kdim = encoder_kdim
        self.encoder_vdim = encoder_vdim
        self.decoder_kdim = decoder_kdim
        self.decoder_vdim = decoder_vdim

        self.positional_encoding = PositionalEncoding(d_model, max_length)

        # initialize the encoder
        if num_encoder_layers > 0:
            if custom_src_module is not None:
                self.custom_src_module = custom_src_module(d_model)
            self.encoder = TransformerEncoder(
                nhead=nhead,
                num_layers=num_encoder_layers,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
                kdim=self.encoder_kdim,
                vdim=self.encoder_vdim,
            )

        # initialize the decoder
        if num_decoder_layers > 0:
            if custom_tgt_module is not None:
                self.custom_tgt_module = custom_tgt_module(d_model)
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

    def forward(self, **kwags):
        """Users should modify this function according to their own tasks."""
        raise NotImplementedError


class PositionalEncoding(nn.Module):
    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)].clone().detach()


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
    ):
        super().__init__()

        self.self_att = MultiheadAttention(
            nhead=nhead,
            d_model=d_model,
            dropout=dropout,
            kdim=kdim,
            vdim=vdim,
        )

        self.pos_ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output, self_attn = self.self_att(
            src1,
            src1,
            src1,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output = self.pos_ffn(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)

        return output, self_attn


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        attention_type="regularMHA",
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        output = src
        attention_lst = []
        for enc_layer in self.layers:
            output, attention = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
            )
            attention_lst.append(attention)
        output = self.norm(output)

        return output, attention_lst


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False
    ):
        super().__init__()
        self.nhead = nhead

        self.self_attn = MultiheadAttention(
            nhead=nhead,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
        )
        self.mutihead_attn = MultiheadAttention(
            nhead=nhead,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
        )

        self.pos_ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        # normalization layers
        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.norm3 = LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        if self.normalize_before:
            tgt1 = self.norm1(tgt)
        else:
            tgt1 = tgt

        # self-attention over the target sequence
        tgt2, self_attn = self.self_attn(
            query=tgt1,
            key=tgt1,
            value=tgt1,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            pos_embs=pos_embs_tgt,
        )

        # add & norm
        tgt = tgt + self.dropout1(tgt2)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        if self.normalize_before:
            tgt1 = self.norm2(tgt)
        else:
            tgt1 = tgt

        # multi-head attention over the target sequence and encoder states

        tgt2, multihead_attention = self.mutihead_attn(
            query=tgt1,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            pos_embs=pos_embs_src,
        )

        # add & norm
        tgt = tgt + self.dropout2(tgt2)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        if self.normalize_before:
            tgt1 = self.norm3(tgt)
        else:
            tgt1 = tgt

        tgt2 = self.pos_ffn(tgt1)

        # add & norm
        tgt = tgt + self.dropout3(tgt2)
        if not self.normalize_before:
            tgt = self.norm3(tgt)

        return tgt, self_attn, multihead_attention


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        output = tgt
        self_attns, multihead_attns = [], []
        for dec_layer in self.layers:
            output, self_attn, multihead_attn = dec_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_embs_tgt=pos_embs_tgt,
                pos_embs_src=pos_embs_src,
            )
            self_attns.append(self_attn)
            multihead_attns.append(multihead_attn)
        output = self.norm(output)

        return output, self_attns, multihead_attns


class NormalizedEmbedding(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.emb = Embedding(
            num_embeddings=vocab, embedding_dim=d_model, blank_id=0
        )
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)


def get_key_padding_mask(padded_input, pad_idx):
    """Creates a binary mask to prevent attention to padded locations.
    """
    if len(padded_input.shape) == 4:
        bz, time, ch1, ch2 = padded_input.shape
        padded_input = padded_input.reshape(bz, time, ch1 * ch2)

    key_padded_mask = padded_input.eq(pad_idx).to(padded_input.device)

    # if the input is more than 2d, mask the locations where they are silence
    # across all channels
    if len(padded_input.shape) > 2:
        key_padded_mask = key_padded_mask.float().prod(dim=-1).bool()
        return key_padded_mask.detach()

    return key_padded_mask.detach()


def get_lookahead_mask(padded_input):
    """Creates a binary mask for each sequence which maskes future frames.
    """
    seq_len = padded_input.shape[1]
    mask = (
        torch.triu(torch.ones((seq_len, seq_len), device=padded_input.device))
        == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.detach().to(padded_input.device)
