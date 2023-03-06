import coremltools as ct
from typing_extensions import Literal
from typing import Union


CoreMLModelSpec = ct.proto.Model_pb2.Model
MLModel_Or_MLSpec = Union[ct.models.MLModel, CoreMLModelSpec]


__all__ = [
    "find_io_type", "assign_io_types", "find_input_type", "find_output_type",
]


# "google.protobuf.pyext._message.RepeatedCompositeContainer"
def assign_io_types(
    from_: ct.proto.Model_pb2.FeatureDescription,
    to: ct.proto.Model_pb2.FeatureDescription,
) -> None:
    """
    Assigns the type of `from_` to `to`.
    Almost always used in conjunction with `find_io_type`

    Example:
        assign_io_type(
            find_io_type(fpn, key="embedding", search_outputs=True),
            find_io_type(classifier_head, key="embedding", search_inputs=True)
        )
    """
    to.ParseFromString(from_.SerializeToString())


def find_io_type(
    spec: ct.proto.Model_pb2.Model,
    key: str,
    search_inputs: bool = False,
    search_outputs: bool = False,
) -> ct.proto.Model_pb2.FeatureDescription:
    """
    Fetches a specific input / output node from `spec.description.{input|output}`
    You control whether to search inputs or outputs with the `search_{inputs|outputs}` flag
      and can only use one when calling the function
    """
    if not sum([search_inputs, search_outputs]) == 1:
        raise TypeError(
            f"You need to set any _one_ of `search_inputs` or `search_outputs` to True"
        )
    if search_inputs:
        return _find_type(spec, key, "input")
    if search_outputs:
        return _find_type(spec, key, "output")


def _find_type(
    spec, key, input_or_output: Literal["input", "output"]
) -> ct.proto.Model_pb2.FeatureDescription:
    for item in getattr(spec.description, input_or_output):
        if item.name == key:
            return item
    raise KeyError(f"Key '{key}' not found in spec's {input_or_output}s")


def find_input_type(
    spec: ct.proto.Model_pb2.Model, key: str
) -> ct.proto.Model_pb2.FeatureDescription:
    return find_io_type(spec, key, search_inputs=True)


def find_output_type(
    spec: ct.proto.Model_pb2.Model, key: str
) -> ct.proto.Model_pb2.FeatureDescription:
    return find_io_type(spec, key, search_outputs=True)


def spec(m: MLModel_Or_MLSpec):
    return m.get_spec() if isinstance(m, ct.models.MLModel) else m


def get_input_feature_names(mlmod: MLModel_Or_MLSpec):
    return [inp.name for inp in spec(mlmod).description.input]


def get_output_feature_names(mlmod: MLModel_Or_MLSpec):
    return [out.name for out in spec(mlmod).description.output]
