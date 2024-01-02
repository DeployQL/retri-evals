from mteb import AbsTaskRetrieval, MSMARCOv2 as mtebMSMARCOv2, CQADupstackEnglishRetrieval as mtebCQA
from reps.evaluation.retriever import DenseRetriever
import logging
from time import time
from mteb.abstasks import DRESModel

logger = logging.getLogger(__name__)

INDEX_KEY = "indexer"
DOC_PROCESSING_KEY = "doc_processor"
QUERY_PROCESSING_KEY = "query_processor"

class IndexedTask:

    def evaluate(
            self,
            model,
            split="test",
            batch_size=128,
            corpus_chunk_size=None,
            score_function="cos_sim",
            **kwargs
    ):
        try:
            from beir.retrieval.evaluation import EvaluateRetrieval
        except ImportError:
            raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")

        if not self.data_loaded:
            self.load_data()

        corpus, queries, relevant_docs = self.corpus[split], self.queries[split], self.relevant_docs[split]
        model = model if self.is_dres_compatible(model) else DRESModel(model)

        retriever = EvaluateRetrieval(model, score_function=score_function)  # or "cos_sim" or "dot"
        start_time = time()
        results = retriever.retrieve(corpus, queries)
        end_time = time()
        logger.info("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

        ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, retriever.k_values, ignore_identical_ids=kwargs.get("ignore_identical_ids", True))
        mrr = retriever.evaluate_custom(relevant_docs, results, retriever.k_values, "mrr")

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }

        return scores


class MSMARCOv2(IndexedTask, mtebMSMARCOv2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CQADupstackEnglishetrieval(IndexedTask, mtebCQA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)