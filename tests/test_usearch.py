import numpy as np
import uuid

from retri_eval.indexes.indexing import IndexingDocument
from retri_eval.indexes.usearch_index import USearchIndex, USearchDocument


class TestUSearch:
    def test_index(self):
        index = USearchIndex(
            "test",
        )
        assert index is not None, "failed to create qdrant index"

    def test_index_writes(self):
        index = USearchIndex(
            "test",
            dims=3,
        )
        index.add(
            USearchDocument(
                id=uuid.uuid4().hex,
                doc_id="123",
                text="some test text",
                embedding=np.asarray([0.1, 1.0, 2.0]),
            )
        )
        out = index.search(vector=np.asarray([0.1, 1.0, 2.0]), limit=2)
        assert len(out) == 1, "failed to search data in the index."
