import json

from sentence_transformers import SentenceTransformer

from retri_eval.evaluation.mteb_tasks import CQADupstackEnglishRetrieval, Touche2020
from retri_eval.evaluation.retriever import DenseRetriever
from retri_eval.indexes.qdrant_index import QdrantIndex, QdrantDocument
from retri_eval.processing.basic_query_processor import QueryProcessor
from retri_eval.processing.beir_title_processor import BeirTitleProcessor
from qdrant_client.models import VectorParams, Distance
from mteb import MTEB



class HFModel:
    def __init__(self, model):
        self.model = model

    def encode(self, batch, **kwargs):
        return self.model.encode(batch)

    def encode_queries(self, batch, **kwargs):
        return self.model.encode([f"Represent this sentence for searching relevant passages: {query}" for query in batch])


if __name__ == "__main__":
    model_name = "BAAI/bge-small-en-v1.5"
    model = SentenceTransformer(model_name)

    index = QdrantIndex(
        "Touche", vector_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    doc_processor = BeirTitleProcessor(model, name=model_name)
    query_processor = QueryProcessor(model, name=model_name)

    retriever = DenseRetriever(
        index=index,
        query_processor=query_processor,
        doc_processor=doc_processor,
    )

    id = f"{doc_processor.id}-{query_processor.id}"
    print(f"evaluation id: {id}")

    eval = MTEB(tasks=[Touche2020()], task_langs=["en"])

    results = eval.run(
        retriever,
        verbosity=2,
        overwrite_results=True,
        output_folder=f"results/{id}",
        eval_splits=["test"],
    )

    print(json.dumps(results, indent=1))
