import unittest
import uuid
from typing import List

from python.sdk.indexes.qdrant_index import QdrantIndex, QdrantDocument
from pydantic import BaseModel, ConfigDict



class TestQdrant(unittest.TestCase):
    def test_index(self):
        index = QdrantIndex("test", path=None, size=3)
        self.assertIsNotNone(index)

    def test_index_writes(self):
        index = QdrantIndex("test", path=None, size=3)
        index.add(QdrantDocument(
            id=uuid.uuid4().hex,
            text="some test text",
            embedding=[0.1, 1., 2.]
        ))
        out = index.search(vector=[0.1,1.,2.], limit=2)
        self.assertEqual(1, len(out))

