import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Iterable
from whisper.model import AudioEncoder
from .resblock import ResidualAttentionBlockANE
from .layer_norm import LayerNormANE


class AudioEncoderANE(AudioEncoder):
    @classmethod
    def from_stock_encoder(cls, encoder: AudioEncoder):
        self = cls(
            n_mels = encoder.conv1.in_channels,
            n_ctx = encoder.positional_embedding.shape[0],
            n_state = encoder.conv1.out_channels,
            n_head = encoder.blocks[0].attn.n_head,
            n_layer = len(encoder.blocks),
        )

        self.conv1.load_state_dict(encoder.conv1.state_dict())
        self.conv2.load_state_dict(encoder.conv2.state_dict())

        # (seq_len, dim) -> (dim, 1, seq_len)
        self.positional_embedding.copy_(encoder.positional_embedding)
        self.positional_embedding = self.positional_embedding.permute(1, 0).unsqueeze(1)

        # Override
        self.blocks: Iterable[ResidualAttentionBlockANE] = nn.ModuleList()
        for block in encoder.blocks:
            self.blocks.append(ResidualAttentionBlockANE.from_stock_block(block))

        self.ln_post = LayerNormANE(encoder.conv1.out_channels)
        self.ln_post.load_state_dict(encoder.ln_post.state_dict())

        return self

    def forward(self, x: Tensor) -> Tensor:
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        # This was in the original implementation. Not needed in ANE version
        # x = x.permute(0, 2, 1)
        # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"

        # Add dummy dimension that makes it a 4D tensor that is ready for conv layers
        x = x.unsqueeze(2)

        x = (x + self.positional_embedding) # .to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x
