import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Iterable
from ane_transformers.huggingface.distilbert import LayerNormANE as _LayerNormANE
from copy import deepcopy


__all__ = ["MultiHeadAttentionANE"]


class MultiHeadAttentionANE(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Conv2d(n_state, n_state, 1)
        self.key = nn.Conv2d(n_state, n_state, 1, bias=False)
        self.value = nn.Conv2d(n_state, n_state, 1)
        self.out = nn.Conv2d(n_state, n_state, 1)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        # kv_cache: Optional[dict] = None,
    ) -> Tensor:
        """
        Parameters:
            x:  (bs, dim, 1, seg_len)
            xa: (bs, dim, 1, seg_len)

        Returns:
            `torch.Tensor` of shape `(bs, dim, 1, seq_len)`
        """
        verbose = False

        assert x.ndim == 4, f"Expected 4D input of shape (bs, dim, 1, seg_len), got {x.shape} instead"
        assert x.shape[2] == 1, f"Expected third dim to be 1. Got {x.shape[2]} instead (full shape -> {x.shape})."

        if xa is not None:
            assert xa.ndim == 4, f"Expected 4D input of shape (bs, dim, 1, seg_len), got {x.shape} instead"
            assert xa.shape[2] == 1, f"Expected third dim to be 1. Got {x.shape[2]} instead (full shape -> {x.shape})."

            if verbose:
                print("`x`  shape: ", x.shape)
                print("`xa` shape: ", xa.shape)

        bs, dim, dummy, seqlen = x.size()

        q = self.query(x)
        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)

        # print(f"Input q,k,v shapes = {q.shape, k.shape, v.shape}")

        if verbose:
            print("Query shape: ", q.shape)
            print("Key shape:   ", k.shape)
            print("Value shape: ", v.shape)

        # Validate mask -- copied from distilbert
        # if mask is not None:
        #     expected_mask_shape = [bs, seqlen, 1, 1]
        #     if mask.dtype == torch.bool:
        #         mask = mask.logical_not().float() * -1e4
        #     elif mask.dtype == torch.int64:
        #         mask = (1 - mask).float() * -1e4
        #     elif mask.dtype != torch.float32:
        #         raise TypeError(f"Unexpected dtype for mask: {mask.dtype}")

        #     if len(mask.size()) == 2:
        #         # equivalent to `mask = mask[..., None, None]`
        #         # adds 2 leading dimensions required for broadcasting
        #         mask = mask.unsqueeze(2).unsqueeze(2)

        #     if list(mask.size()) != expected_mask_shape:
        #         raise RuntimeError(
        #             f"Invalid shape for `mask` (Expected {expected_mask_shape}, got {list(mask.size())}"
        #         )

        if mask is not None:
            mask = mask[:seqlen, :seqlen]
            mask = mask.flip(0,1).unsqueeze(1).unsqueeze(0)

            assert list(mask.size()) == [bs, seqlen, 1, seqlen]

        # qkv_attention
        dim_per_head = dim // self.n_head
        """
        NOTE: Originally, I'd set the value to ** -0.25 because that is what is originally implemented
        in the `whisper` codebase. See this line specifically:
            https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/model.py#L90

        However, when running empirical tests, -0.25 caused heavy mismatch between the two modules, whereas
        -0.5 worked correctly. I am lost as to why this is, but it is correct, and interestingly, also the default
        value in apple's implementation as seen in these two places:
            https://github.com/apple/ml-ane-transformers/blob/da64000fa56cc85b0859bc17cb16a3d753b8304a/ane_transformers/reference/multihead_attention.py#L48
            https://github.com/apple/ml-ane-transformers/blob/da64000fa56cc85b0859bc17cb16a3d753b8304a/ane_transformers/huggingface/distilbert.py#L161
        """
        # normalize_factor = float(dim_per_head) ** -0.25
        normalize_factor = float(dim_per_head) ** -0.5

        mh_q = q.split(
            dim_per_head,
            dim=1)  # (bs, dim_per_head, 1, max_seq_length) * n_heads
                    # (b, c, h, q)
        mh_k = k.transpose(1, 3).split(
            dim_per_head,
            dim=3)  # (bs, max_seq_length, 1, dim_per_head) * n_heads
                    # (b, k, h, c)
        mh_v = v.split(
            dim_per_head,
            dim=1)  # (bs, dim_per_head, 1, max_seq_length) * n_heads
                    # (b, c, h, k)

        if verbose:
            print(f"`mh_q` ({len(mh_q)}) shape: ", mh_q[0].shape)
            print(f"`mh_k` ({len(mh_k)}) shape: ", mh_k[0].shape)
            print(f"`mh_v` ({len(mh_v)}) shape: ", mh_v[0].shape)

        # `qk = q @ k``
        attn_weights = [
            torch.einsum('bchq,bkhc->bkhq', [qi, ki]) * normalize_factor
            for qi, ki in zip(mh_q, mh_k)
        ]  # (bs, max_seq_length, 1, max_seq_length) * n_heads

        if verbose:
            print(f"`attn_weights` ({len(attn_weights)}) shape: ", attn_weights[0].shape)

        if mask is not None:
            for head_idx in range(self.n_head):
                attn_weights[head_idx] = attn_weights[head_idx] + mask

        # `w = F.softmax(qk.float(), dim=-1)`
        attn_weights = [aw.softmax(dim=1) for aw in attn_weights
                        ]  # (bs, max_seq_length, 1, max_seq_length) * n_heads

        # (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        attn = [
            torch.einsum('bkhq,bchk->bchq', wi, vi)
            for wi, vi in zip(attn_weights, mh_v)
        ]  # (bs, dim_per_head, 1, max_seq_length) * n_heads

        attn = torch.cat(attn, dim=1)  # (bs, dim, 1, max_seq_length)

        return self.out(attn)

        # FIXME: Is the squeeze just slowing us down unnecessarily?
        return self.out(attn).permute(0, 3, 1, 2)
