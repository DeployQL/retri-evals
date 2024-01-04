

import uuid
from typing import List, Dict
import json
from reps.evaluation.mteb_tasks import CQADupstackEnglishRetrieval, Touche2020
from reps.evaluation.retriever import DenseRetriever
from reps.indexes.qdrant_index import QdrantIndex, QdrantDocument
from reps.indexes.indexing import MTEBDocument
from reps.processing.pipeline import ProcessingPipeline, Input, Output
from FlagEmbedding import FlagModel
from qdrant_client.models import VectorParams, Distance
from mteb import MTEB


class DocumentProcessor(ProcessingPipeline[Dict[str, str], QdrantDocument]):
    def __init__(self, model, name='', version=''):
        super().__init__()
        self.model = model

    def process(self, batch: List[Dict[str, str]], batch_size: int=0, **kwargs) -> List[QdrantDocument]:
        """
        Takes a string of a document and returns a document for the index..
        :param batch:
        :param kwargs:
        :return:
        """
        # stubbed for demonstration purposes
        chunker = lambda x: [x]

        results = []
        for x in batch:
            doc = MTEBDocument(**x)

            chunks = chunker(doc.text)
            embedding = self.model.encode(chunks)
            for i, chunk in enumerate(chunks):
                results.append(QdrantDocument(
                    id=uuid.uuid4().hex, # qdrant requires a uuid.
                    doc_id=doc.doc_id,
                    text=chunk,
                    embedding=embedding[i],
                ))
        return results

class QueryProcessor(ProcessingPipeline[str, List[float]]):
    def __init__(self, model, name = '', version = ''):
        super().__init__()
        self.model = model

    def process(self, batch: List[str], batch_size: int=0, **kwargs) -> List[List[float]]:
        return self.model.encode_queries(batch)


if __name__ == "__main__":
    model_name ="BAAI/bge-small-en-v1.5"
    model = FlagModel(model_name,
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                      use_fp16=False)

    index = QdrantIndex("Touche", vector_config=VectorParams(size=384, distance=Distance.COSINE))
    doc_processor = DocumentProcessor(model, name=model_name)
    query_processor = QueryProcessor(model, name=model_name)

    retriever = DenseRetriever(
        index=index,
        query_processor=query_processor,
        doc_processor=doc_processor,
    )

    id = f"{doc_processor.id}-{query_processor.id}"
    print(f"evaluation id: {id}")

    eval = MTEB(tasks=[Touche2020()], task_langs=['en'])

    results = eval.run(retriever, verbosity=2, overwrite_results=True, output_folder=f"results/{id}", eval_splits=['test'])


    print(json.dumps(results, indent=1))