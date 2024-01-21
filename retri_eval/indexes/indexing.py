from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar, Optional, Dict
from typing import TypedDict
from pydantic import BaseModel, ConfigDict
from retri_eval.indexes.numpy_type import NdArray


T = TypeVar("T")
K = TypeVar("K")


class MTEBDocument(BaseModel):
    """
    MTEBDocument is a pydantic model of the fields we expect from MTEB data.

    Note: doc_id is manually added to the documents in our DenseRetriever model.
    """

    doc_id: str
    text: str
    title: str


class SearchResponse(BaseModel):
    """
    SearchResponse returns data. It doesn't make any assumptions of the payload
    returned, it only requires that we have an id and a score.
    """

    id: str
    doc_id: str
    score: float
    text: str


class IndexingDocument(BaseModel):
    """
    IndexingDocument is a Generic document model that takes in the base fields each index should expect to handle.
    """

    id: str
    doc_id: str
    embedding: NdArray
    text: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Index(ABC, Generic[T]):
    """
    Indexes enable storing data. The generic interface doesn't know what type of data, only that we are able to
    add it.

    It seems likely that the search response object will also need to be generic in the future.
    """

    @abstractmethod
    def add(self, item: List[T] | T):
        pass

    @abstractmethod
    def search(
        self,
        vector: List[float],
        limit: Optional[int] = 0,
        fields: Optional[List[str]] = None,
    ) -> List[SearchResponse]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def save(self):
        pass
