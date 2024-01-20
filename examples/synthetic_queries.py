import uuid
from typing import List, Dict
import json
from retri_eval.evaluation.mteb_tasks import HotpotQA
from retri_eval.evaluation.retriever import DenseRetriever
from retri_eval.indexes.qdrant_index import QdrantIndex, QdrantDocument
from retri_eval.indexes.indexing import MTEBDocument
from retri_eval.indexes.usearch_index import USearchIndex, USearchDocument
from retri_eval.processing.pipeline import ProcessingPipeline, Input, Output
from FlagEmbedding import FlagModel
from retri_eval.datasets.types import DataPair
from qdrant_client.models import VectorParams, Distance
from mteb import MTEB
from retri_eval.datasets.training_data import TrainingDataManager
from tqdm import tqdm


class DocumentProcessor(ProcessingPipeline[Dict[str, str], QdrantDocument]):
    def __init__(self, model, name="", version=""):
        super().__init__(name, version)
        self.model = model

    def process(
        self, batch: List[Dict[str, str]], batch_size: int = 0, **kwargs
    ) -> List[QdrantDocument]:
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
                results.append(
                    QdrantDocument(
                        id=uuid.uuid4().hex,  # qdrant requires a uuid.
                        doc_id=doc.doc_id,
                        text=chunk,
                        embedding=embedding[i],
                    )
                )
        return results


class QueryProcessor(ProcessingPipeline[str, List[float]]):
    def __init__(self, model, name="", version=""):
        super().__init__(name, version)
        self.model = model

    def process(
        self, batch: List[str], batch_size: int = 0, **kwargs
    ) -> List[List[float]]:
        return self.model.encode_queries(batch)


class TolerantModel:
    def __init__(self, model):
        self.model = model

    def encode(self, batch, **kwargs):
        return self.model.encode(batch)


if __name__ == "__main__":
    task = HotpotQA()
    split = 'test'
    task.load_data()
    # data = Dict[str, Dict[str, str]]
    data = task.corpus[split]
    # queries = Dict[str, str]
    queries = task.queries[split]
    rel = task.relevant_docs[split]

    sample_keys = list(data.keys())[:5]
    sd = [data[x] for x in sample_keys]

    model_name = "BAAI/bge-small-en-v1.5"
    model = FlagModel(
        model_name,
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        use_fp16=False,
    )
    index_name = "results/synthetic_queries/hotpot"
    index = USearchIndex(name=index_name, dims=384)
    doc_processor = DocumentProcessor(model, name=model_name)
    query_processor = QueryProcessor(model, name=model_name)

    # create an index
    if index.count() == 0:
        for i, (key, doc) in enumerate(tqdm(data.items())):
            if i > 15000 and key not in sample_keys:
                continue
            doc_dict = dict(doc)
            doc_dict['doc_id'] = key
            processed = doc_processor.process([doc_dict])[0]
            # below is just a hack to convert between document types.
            usearch_doc = USearchDocument(**dict(processed))
            index.add(usearch_doc)

        index.save()

    mngr = TrainingDataManager(
        index=index,
        query_processor=query_processor,
        doc_processor=doc_processor,
    )

    formatted_data = [
        DataPair(doc=x['text'], query='', doc_id=sample_keys[i], is_positive=False) for i,x in enumerate(sd)
    ]

    labeled = mngr.process_pairs(formatted_data)
    print(labeled)

    # print(sq)
    # sr = [rel[x] for x in sample_keys]
    # print(sr)
    #
    #

    #
    # retriever = DenseRetriever(
    #     index=index,
    #     query_processor=query_processor,
    #     doc_processor=doc_processor,
    # )
    #
    # id = f"{doc_processor.id}-{query_processor.id}"
    # print(f"evaluation id: {id}")
    #
    # eval = MTEB(tasks=[], task_langs=["en"])
    #
    # results = eval.run(
    #     retriever,
    #     verbosity=2,
    #     overwrite_results=True,
    #     output_folder=f"results/{id}",
    #     eval_splits=["test"],
    # )
    #
    # print(json.dumps(results, indent=1))
