from python.sdk.indexes.indexing import Index, SearchResponse
from typing import Any, get_type_hints, Generic, TypeVar, Optional, List, Mapping
from pydantic import BaseModel
import builtins
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct

T = TypeVar("T")

class QdrantDocument(BaseModel):
    id: str
    embedding: List[float]
    text: str

class QdrantIndex(Index):
    def __init__(
            self,
            name: str,
            path: Optional[str],
            embedding_key: str = "embedding",
            id_key: str = "id",
            vector_config = Mapping[str, VectorParams] | VectorParams,
    ):
        self.name = name
        # self.index = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
        self.index = QdrantClient(path if path else ":memory:")
        self.index.recreate_collection(
            name,
            vectors_config=vector_config,
        )
        self.embedding_key = embedding_key
        self.id_key = id_key

    def add(self, item: QdrantDocument):
        if isinstance(item, list):
            points = []
            for x in item:
                points.append(PointStruct(
                    id=x.id,
                    vector=dict(x).pop(self.embedding_key),
                    payload=x,
                ))
            self.index.upsert(
                collection_name=self.name,
                wait=True,
                points=points,
            )
        else:
            self.index.upsert(
                collection_name=self.name,
                wait=True,
                points=[
                    PointStruct(
                        id=item.id,
                        vector=dict(item).pop(self.embedding_key),
                        payload=item,
                    )
                ],
            )

    def search(self, query: Optional[str]=None, vector:Optional[List[float]] = None, limit: Optional[int]=0, fields:Optional[List[str]]=None) -> List[SearchResponse]:
        if not query and not vector:
            raise ValueError("query or vector must be provided.")

        if query and vector:
            raise ValueError("must provide either query or vector, not both.")

        if vector:
            return self.index.search(
                collection_name=self.name,
                query_vector=vector,
            )

        return self.index.query(
            collection_name=self.name,
            query_text=query,
        )

    def count(self) -> int:
        return 0

