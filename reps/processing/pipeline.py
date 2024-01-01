from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic

import torch

Input = TypeVar("Input")
Output = TypeVar("Output")


class EmbeddedOutput:
    """
    EmbeddedOutput is a standard output for document embeddings.
    """
    id: str
    embedding: torch.Tensor

class ProcessingPipeline(ABC, Generic[Input, Output]):
    def __init__(self, name: str='', version:str=''):
        self.name = name
        self.version = version

    @property
    def id(self) -> str:
        """
        Creates a unique id for this pipeline.
        :return:
        """
        return hash(self.name+self.version)

    @abstractmethod
    def process(self, batch: List[Input], **kwargs) -> List[Output]:
        """

        :param batch: a list of strings. Strings could be urls, file paths, or raw text.
        :param kwargs: other kwargs that can be passed into the pipeline.
        :return: List[tensor]
        """
        pass

