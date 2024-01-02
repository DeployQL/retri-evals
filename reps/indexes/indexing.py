from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar, Optional, Dict
from typing import TypedDict
from pydantic import BaseModel

T = TypeVar("T")
K = TypeVar("K")

class MTEBDocument(BaseModel):
    """
    MTEBDocument is a pydantic model of the fields we expect from MTEB data.

    Note: doc_id is manually added to the documents in our DenseRetriever model.
    """
    doc_id : str
    text: str
    title: str

class SearchResponse(TypedDict):
    """
    SearchResponse returns data. It doesn't make any assumptions of the payload
    returned, it only requires that we have an id and a score.
    """
    id: str
    doc_id: str
    score: float

class Index(ABC, Generic[T]):

    @abstractmethod
    def add(self, item: List[T] | T):
        pass

    @abstractmethod
    def search(self, query: List[float], limit: Optional[int]=0, fields:Optional[List[str]]=None) -> List[SearchResponse]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass
