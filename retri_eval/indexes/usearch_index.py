from retri_eval.indexes.indexing import Index, SearchResponse
from typing import Any, get_type_hints, Generic, TypeVar, Optional, List, Mapping
from pydantic import BaseModel, ConfigDict
from usearch.index import Index as USIndex
import dbm
import os
import numpy as np

from retri_eval.indexes.numpy_type import NdArray

T = TypeVar("T")


class USearchDocument(BaseModel):
    id: str
    doc_id: str
    embedding: NdArray
    text: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class USearchIndex(Index[USearchDocument]):
    def __init__(
            self,
            name: str,
            location: Optional[str] = ":memory:",
            embedding_key: str = "embedding",
            id_key: str = "id",
            dims: int = 384,
            multi=False,
            read_only=False,
    ):
        self.name = name

        if not os.path.exists(f"{name}/index") and not read_only:
            os.makedirs(f"{name}", exist_ok=True)
            self.index = USIndex(
                ndim=dims,
                dtype='f16',
                multi=False,
            )

        # self.index = self._index.view(f"{name}/index")
        else:
            self.index = USIndex.restore(f"{name}/index", view=read_only)


        self.text = dbm.open(f"{name}/text", 'c')
        # usearch requires int keys. we map between any given id to an int.
        self.keys = dbm.open(f"{name}/keys", 'c')
        self.id_allocated = len(self.keys)

        self.embedding_key = embedding_key
        self.id_key = id_key

    def add(self, item: USearchDocument | List[USearchDocument]):
        if isinstance(item, list):
            points = []
            for x in item:
                doc_dict = dict(x)
                id = doc_dict.pop(self.id_key)
                embedding = doc_dict.pop(self.embedding_key)

                allocated_id = self.id_allocated + 1
                self.index.add(allocated_id, embedding)
                self.keys[str(allocated_id)] = id
                self.text[str(allocated_id)] = x.text

                self.id_allocated += 1

        else:
            doc_dict = dict(item)
            id = doc_dict.pop(self.id_key)

            allocated_id = self.id_allocated + 1
            embedding = doc_dict.pop(self.embedding_key)

            self.index.add(allocated_id, embedding)
            self.keys[str(allocated_id)] = id
            self.text[str(allocated_id)] = item.text

            self.id_allocated += 1

    def search(
            self,
            vector: List[float],
            limit: Optional[int] = 0,
            fields: Optional[List[str]] = None,
    ) -> List[SearchResponse]:
        results = self.index.search(vector, limit if limit else 100
        )
        print([x.key for x in results])
        print([self.text.get(str(x.key), '') for x in results])
        return [
            SearchResponse(id=str(r.key), doc_id=str(self.keys[str(r.key)]), score=r.distance, text=self.text.get(str(r.key), ''))
            for r in results
        ]

    def count(self) -> int:
        return len(self.text)

    def save(self):
        self.index.save(f"{self.name}/index")
        self.text.sync()
        self.keys.sync()
