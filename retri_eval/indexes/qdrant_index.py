from retri_eval.indexes.indexing import Index, SearchResponse
from typing import Any, get_type_hints, Generic, TypeVar, Optional, List, Mapping
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct

T = TypeVar("T")


class QdrantDocument(BaseModel):
    id: str
    doc_id: str
    embedding: List[float]
    text: str

class QdrantIndex(Index[QdrantDocument]):
    def __init__(
            self,
            name: str,
            location: Optional[str]=":memory:",
            embedding_key: str = "embedding",
            id_key: str = "id",
            vector_config = Mapping[str, VectorParams] | VectorParams,
    ):
        self.name = name

        self.index = QdrantClient(location)
        self.index.recreate_collection(
            name,
            vectors_config=vector_config,
        )
        self.embedding_key = embedding_key
        self.id_key = id_key

    def add(self, item: QdrantDocument | List[QdrantDocument]):
        if isinstance(item, list):
            points = []
            for x in item:
                doc_dict = dict(x)

                points.append(PointStruct(
                    id=doc_dict.pop(self.id_key),
                    vector=doc_dict.pop(self.embedding_key),
                    payload=doc_dict,
                ))
            self.index.upsert(
                collection_name=self.name,
                wait=True,
                points=points,
            )
        else:
            doc_dict = dict(item)
            self.index.upsert(
                collection_name=self.name,
                wait=True,
                points=[
                    PointStruct(
                        id=doc_dict.pop(self.id_key),
                        vector=doc_dict.pop(self.embedding_key),
                        payload=doc_dict,
                    )
                ],
            )

    def search(self, vector: List[float], limit: Optional[int]=0, fields:Optional[List[str]]=None) -> List[SearchResponse]:
        results = self.index.search(
            collection_name=self.name,
            query_vector=vector,
            limit=limit if limit else 0,
        )

        return [SearchResponse(id=r.id, doc_id=r.payload['doc_id'], score=r.score) for r in results]

    def count(self) -> int:
        return self.index.count(self.name).count

