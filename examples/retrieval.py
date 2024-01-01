

import uuid
from typing import List

from pydantic import BaseModel

from python.sdk.evaluation.mteb import MSMARCOv2
from python.sdk.indexes.qdrant_index import QdrantIndex, QdrantDocument
from python.sdk.processing.pipeline import ProcessingPipeline, Input, Output
from FlagEmbedding import FlagModel

from mteb import MTEB

class Document(BaseModel):
    id: str
    doc_id: str
    text: str
    title: str


class DocumentProcessor(ProcessingPipeline[Document, List[QdrantDocument]]):
    def process(self, batch: List[Document], **kwargs) -> List[QdrantDocument]:
        """
        Takes a string of a document and returns an embedding.
        :param batch:
        :param kwargs:
        :return:
        """
        chunker = lambda x: x.split(" ")
        embedder = lambda x: list(range(0, 5))

        results = []
        for doc in batch:
            chunks = chunker(doc.text)
            for chunk in chunks:
                results.append(QdrantDocument(
                    id=uuid.uuid4().hex,
                    doc_id=doc.id,
                    text=chunk,
                    embedding=embedder(chunk),
                ))
        return results

if __name__ == "__main__":
    model = FlagModel("BAAI/bge-small-en-v1.5",
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                      use_fp16=True)
    index = QdrantIndex("msmarco", size=5)
    processing = DocumentProcessor()
    eval = MTEB(tasks=[MSMARCOv2()])
    results = eval.run(model, verbosity=2, indexer=index, processor=processing)

    print(results)