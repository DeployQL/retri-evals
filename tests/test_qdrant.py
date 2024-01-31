import numpy as np
import uuid

from retri_eval.indexes.qdrant_index import QdrantIndex, QdrantDocument
from qdrant_client.models import VectorParams, Distance


class TestQdrant:
    def test_index(self):
        index = QdrantIndex(
            "test", vector_config=VectorParams(size=3, distance=Distance.COSINE)
        )
        assert index is not None, "failed to create qdrant index"

    def test_index_writes(self):
        index = QdrantIndex(
            "test", vector_config=VectorParams(size=3, distance=Distance.COSINE)
        )
        index.add(
            QdrantDocument(
                id=uuid.uuid4().hex,
                doc_id="123",
                text="some test text",
                embedding=np.asarray([0.1, 1.0, 2.0]),
            )
        )
        out = index.search(vector=[0.1, 1.0, 2.0], limit=2)
        assert len(out) == 1, "failed to search data in the index."
