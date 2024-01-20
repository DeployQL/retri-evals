from typing_extensions import Annotated
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer
import numpy as np

def nd_array_custom_before_validator(x):
    return x


def nd_array_custom_serializer(x):
    return str(x)

NdArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=str),
]