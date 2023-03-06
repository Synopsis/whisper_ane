# Copied over largely from ane_transformers.huggingface.distilbert
from ane_transformers.reference.layer_norm import LayerNormANE as _LayerNormANE


__all__ = ["LayerNormANE"]


# Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
# apply scale and bias terms in opposite orders. In order to accurately restore a
# state_dict trained using the former into the the latter, we adjust the bias term
def correct_for_bias_scale_order_inversion(
    state_dict, prefix, local_metadata, strict, missing_keys,
    unexpected_keys, error_msgs,
):
    state_dict[prefix + "bias"] = (
        state_dict[prefix + "bias"] / state_dict[prefix + "weight"]
    )
    return state_dict


class LayerNormANE(_LayerNormANE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(correct_for_bias_scale_order_inversion)

    def forward(self, inputs):
        """
        Parameters:
            inputs: tensor of shape `(bs, dim, 1, seq_len)`

        Returns:
            tensor of shape `(bs, dim, 1, seq_len)`
        """
        return super().forward(inputs)
