import torch.nn as nn
import torch
import whisper

from copy import deepcopy
from .multihead_attention import MultiHeadAttentionANE


__all__ = ["transfer_lin_wts_to_conv", "transfer_attn_block_wts_to_optimised_block"]


def transfer_lin_wts_to_conv(linear: nn.Linear, conv: nn.Conv2d) -> nn.Conv2d:
    if linear.bias is not None:
        assert conv.bias is not None, f"`linear` layer has a bias but the `conv` layer was initialised without one"
        conv.bias = deepcopy(linear.bias)
        # conv.bias.copy_(linear.bias)

    assert linear.in_features == conv.in_channels, f"num. input channels do not match"
    assert linear.out_features == conv.out_channels, f"num. output channels do not match"
    assert conv.kernel_size == (1, 1), f"`conv` layer expects > 1D inputs, expected a kernel size of 1"

    # https://discuss.pytorch.org/t/how-to-convert-fully-connected-layer-to-fully-convolutional-layer/97598/2
    with torch.no_grad():
        conv.weight.copy_(linear.weight[:, :, None, None])

    return conv


def transfer_attn_block_wts_to_optimised_block(
    stock_mha: whisper.model.MultiHeadAttention, optim_mha: MultiHeadAttentionANE
) -> MultiHeadAttentionANE:
    # Transfer weights from stock -> ANE optimised layer
    optim_mha.query = transfer_lin_wts_to_conv(stock_mha.query, optim_mha.query)
    optim_mha.key   = transfer_lin_wts_to_conv(stock_mha.key, optim_mha.key)
    optim_mha.value = transfer_lin_wts_to_conv(stock_mha.value, optim_mha.value)
    optim_mha.out   = transfer_lin_wts_to_conv(stock_mha.out, optim_mha.out)

    return optim_mha
