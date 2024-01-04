from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic
import hruid
import torch

Input = TypeVar("Input")
Output = TypeVar("Output")

generator = hruid.Generator()


class EmbeddedOutput:
    """
    EmbeddedOutput is a standard output for document embeddings.
    """
    id: str
    embedding: torch.Tensor

class ProcessingPipeline(ABC, Generic[Input, Output]):
    """
    ProcessingPipelines are a high level abstraction for translating an input to an output.

    This abstraction is used to encapsulate the id behavior of the pipelines.
    """
    def __init__(self, name: str='', version:str=''):
        self.name = name if name else generator.random()
        self.version = version if version else 'v0.0'

    @property
    def id(self) -> str:
        """
        Creates a unique id for this pipeline. This id will be a positive integer that we convert to a string.
        :return:
        """
        return f"{self.name}-{self.version}"

    @abstractmethod
    def process(self, batch: List[Input], batch_size: int=0, **kwargs) -> List[Output]:
        """

        :param batch: a list of strings. Strings could be urls, file paths, or raw text.
        :param kwargs: other kwargs that can be passed into the pipeline.
        :return: List[tensor]
        """
        pass

