from python.sdk.indexes.indexing import Index, SearchResponse
from typing import Any, get_type_hints, Generic, TypeVar, Type, Dict, Optional, List, get_origin
from pydantic import BaseModel
import builtins
import tantivy
from datetime import datetime
import typing

T = TypeVar('T')

DEFAULT_LIMIT = 100

class TantivyIndex(Index, Generic[T]):
    def __init__(self, schema, path: Optional[str]):
        self.schema = schema
        self.index = get_tantivy_index(schema, path)
        self.fields = list(get_fields_annotations(schema).keys())

    def add(self, item: List[T] | T):
        writer = self.index.writer()

        if isinstance(item, list):
            for i in item:
                writer.add_document(tantivy.Document(
                    **(dict(i)),
                ))
        else:
            writer.add_document(tantivy.Document(
                **(dict(item)),
            ))
        writer.commit()

    def search(self, q: str, limit: Optional[int]=0, fields:Optional[List[str]]=None):
        self.index.reload()
        searcher = self.index.searcher()

        query = self.index.parse_query(q, fields if fields else self.fields)
        # returns (score, address) tuples
        hits = searcher.search(query, limit=limit if limit != 0 else DEFAULT_LIMIT).hits

        results = []
        for hit in hits:
            doc = searcher.doc(hit[1])
            doc.add_float("score", float(hit[0]))
            results.append(doc)

        return results

    def count(self) -> int:
        return 0



def get_fields_annotations(m: Type[BaseModel]) -> Dict[str, Any]:
    default_annotations = get_type_hints(m)
    return {field_name: default_annotations[field_name] for field_name in m.__fields__}

def get_tantivy_index(m: Type[BaseModel], path: Optional[str]) -> tantivy.Index:
    annotations = get_fields_annotations(m)

    schema_builder = tantivy.SchemaBuilder()
    for field, annotation in annotations.items():
        if field == "id":
            schema_builder.add_text_field(field, stored=True)
            continue

        if annotation == builtins.str:
            schema_builder.add_text_field(field, stored=True)
        elif annotation == builtins.int:
            schema_builder.add_integer_field(field, stored=True)
        elif annotation == datetime:
            schema_builder.add_date_field(field, stored=True)
        elif annotation == typing.List[str]:
            schema_builder.add_text_field(field, stored=True)
        else:
            raise ValueError(f"type {annotation} not supported")

    schema = schema_builder.build()
    index = tantivy.Index(schema, path=path)
    return index
