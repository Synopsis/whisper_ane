import torch.nn as nn
import torch.nn.functional as F
import torch

from copy import deepcopy
from torch import Tensor
from typing import Iterable
from whisper.model import TextDecoder
from .multihead_attention import MultiHeadAttentionANE
from .resblock import ResidualAttentionBlockANE
from .layer_norm import LayerNormANE


__all__ = ["TextDecoderANE"]


class TextDecoderANE(TextDecoder):
    # def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
    #     super().__init__()

    #     self.token_embedding = nn.Embedding(n_vocab, n_state)
    #     self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

    #     self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
    #         [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
    #     )
    #     self.ln = LayerNorm(n_state)

    @classmethod
    def from_stock_decoder(cls, decoder: TextDecoder):
        n_vocab, n_state = decoder.token_embedding.weight.shape
        self = cls(
            n_vocab = n_vocab,
              n_ctx = decoder.positional_embedding.shape[0],
            n_state = n_state,
             n_head = decoder.blocks[0].attn.n_head,
            n_layer = len(decoder.blocks),
        )

        self.token_embedding.load_state_dict(decoder.token_embedding.state_dict())
        self.positional_embedding = deepcopy(decoder.positional_embedding)

        # Remove mask as an internal attribute. We want this to be a param in the `.forward()` method
        # self._non_persistent_buffers_set.remove("mask")
        # del self.mask

        # Override
        self.blocks: Iterable[ResidualAttentionBlockANE] = nn.ModuleList()
        for block in decoder.blocks:
            self.blocks.append(ResidualAttentionBlockANE.from_stock_block(block))

        self.ln = LayerNormANE(n_state)
        self.ln.load_state_dict(decoder.ln.state_dict())

        return self

    # def forward(self, x: Tensor, xa: Tensor, mask: Tensor):
    def forward(self, x: Tensor, xa: Tensor):
        # offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        offset = 0

        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        # x = x.to(xa.dtype)

        # BSC -> BC1S (most conducive to ANE)
        x = x.transpose(1, 2).unsqueeze(2)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask)

        x = self.ln(x)

        # Permute back to original model's shape to enable matmul
        # NOTE: Is it possible to optimise this via convs? Doubt so.
        x = x.permute(0, 3, 1, 2).squeeze(-1)

        # Removed the `.to()` call as it's irrelevant... right?
        logits = (x @ torch.transpose(self.token_embedding.weight, 0, 1)).float()

        return logits
