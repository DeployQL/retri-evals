import uuid
from typing import List

from retri_eval.indexes.indexing import IndexingDocument
from ir_datasets.datasets.beir import BeirTitleDoc

from retri_eval.processing.pipeline import ProcessingPipeline


class BeirTitleProcessor(ProcessingPipeline[BeirTitleDoc, IndexingDocument]):
    """
    BeirTitleProcessor is a pipeline that takes in a BeirTitleDoc and returns an IndexingDocument.

    You can figure out what datasets use BeirTitleDoc here: https://ir-datasets.com/
    """

    def __init__(self, model, name="", version=""):
        super().__init__(name, version)
        self.model = model

    def process(
        self, batch: List[BeirTitleDoc], batch_size: int = 0, **kwargs
    ) -> List[IndexingDocument]:
        results = []
        for x in batch:
            results.append(self.process_single(x))

        return results

    def process_single(self, doc: BeirTitleDoc) -> IndexingDocument:
        chunks = [doc.default_text()]
        embedding = self.model.encode(chunks)
        for i, chunk in enumerate(chunks):
            return IndexingDocument(
                id=uuid.uuid4().hex,
                doc_id=doc.doc_id,
                text=chunk,
                embedding=embedding[i],
            )
