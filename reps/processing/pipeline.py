from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic
import hruid
import torch
import ctypes

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
    def __init__(self, name: str='', version:str=''):
        self.name = name if name else generator.random()
        self.version = version if version else 'v0.0'

    @property
    def id(self) -> str:
        """
        Creates a unique id for this pipeline.
        :return:
        """
        return ctypes.c_size_t(hash(self.name+self.version)).value

    @abstractmethod
    def process(self, batch: List[Input], batch_size: int=0, **kwargs) -> List[Output]:
        """

        :param batch: a list of strings. Strings could be urls, file paths, or raw text.
        :param kwargs: other kwargs that can be passed into the pipeline.
        :return: List[tensor]
        """
        pass

