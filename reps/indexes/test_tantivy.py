import unittest
from typing import List

from reps.indexes.tantivy_index import TantivyIndex

from reps.indexes.tantivy_index import get_tantivy_index, get_fields_annotations
from pydantic import BaseModel

class TestSchema(BaseModel):
    id: str
    title: str
    paragraphs: List[str]

class TestTantivy(unittest.TestCase):

    def test_schema(self):
        annotations = get_fields_annotations(TestSchema)
        self.assertEqual(annotations["id"], str)
        self.assertEqual(annotations["title"], str)
        self.assertEqual(annotations["paragraphs"], List[str])

    def test_index(self):
        index = TantivyIndex(TestSchema, path=None)
        self.assertIsNotNone(index)

    def test_index_writes(self):
        index = TantivyIndex[TestSchema](TestSchema, path=None)
        index.add(TestSchema(
            id="testId",
            title="some test title",
            paragraphs=["paragraph1", "paragraph2"]
        ))
        out = index.search("paragraph2", 4)

        # we return lists.
        self.assertEqual(2, len(out[0].to_dict()['paragraphs']))

