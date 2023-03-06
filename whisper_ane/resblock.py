import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List
from ane_transformers.huggingface.distilbert import LayerNormANE as _LayerNormANE
from whisper.model import (
    Linear, ResidualAttentionBlock, MultiHeadAttention, LayerNorm,
    AudioEncoder, TextDecoder, Whisper,
)
from typing import Iterable
from copy import deepcopy

from .multihead_attention import MultiHeadAttentionANE
from .layer_norm import LayerNormANE
from .utils import *


class ResidualAttentionBlockANE(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        # LayerNormClass = LayerNorm
        LayerNormClass = LayerNormANE

        self.attn = MultiHeadAttentionANE(n_state, n_head)
        self.attn_ln = LayerNormClass(n_state)

        self.cross_attn = MultiHeadAttentionANE(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNormClass(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNormClass(n_state)


    @classmethod
    def from_stock_block(cls, block: ResidualAttentionBlock):
        n_state = block.attn.query.in_features
        n_head = block.attn.n_head
        cross_attention = block.cross_attn is not None

        self = cls(n_state, n_head, cross_attention)
        self.attn = transfer_attn_block_wts_to_optimised_block(block.attn, self.attn)
        self.attn_ln.load_state_dict(block.attn_ln.state_dict())

        if self.cross_attn is not None:
            self.cross_attn = transfer_attn_block_wts_to_optimised_block(block.cross_attn, self.cross_attn)
            self.cross_attn_ln.load_state_dict(block.cross_attn_ln.state_dict())

        self.mlp = replace_mlp(block.mlp)
        self.mlp_ln.load_state_dict(block.mlp_ln.state_dict())

        return self


    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        """
        Parameters:
            x:  (bs, dim, 1, seg_len)
            xa: (bs, dim, 1, seg_len)

        Returns:
            `torch.Tensor` of shape `(bs, dim, 1, seq_len)`
        """
        verbose =  not True

        if verbose: print("Init `x` shape           : ", {x.shape})
        x = x + self.attn(self.attn_ln(x), mask=mask)

        if self.cross_attn:
            if verbose: print("Pre cross attn `x` shape : ", {x.shape})
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)

        if verbose: print("Pre mlp `x` shape        : ", {x.shape})
        x = x + self.mlp(self.mlp_ln(x))
        return x

        # x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        # if self.cross_attn:
        #     x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        # x = x + self.mlp(self.mlp_ln(x))
        # return x

    def forward_explicit(self, x, xa, mask):
        self.attn_ln(x)


def replace_mlp(
    mlp: nn.Sequential,
    replace_lin_types: List[nn.Module] = (nn.Linear),
) -> nn.Sequential:
    new_layers = []
    for layer in mlp:
        if isinstance(layer, replace_lin_types):
            new_layer = nn.Conv2d(layer.in_features, layer.out_features, 1)
            new_layer = transfer_lin_wts_to_conv(layer, new_layer)
        else:
            new_layer = deepcopy(layer)
        new_layers.append(new_layer)
    return nn.Sequential(*new_layers)
