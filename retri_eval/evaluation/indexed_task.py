import logging
from time import time
import numpy as np

logger = logging.getLogger(__name__)

INDEX_KEY = "indexer"
DOC_PROCESSING_KEY = "doc_processor"
QUERY_PROCESSING_KEY = "query_processor"


class IndexedTask:
    """
    IndexedTask is a stubbed MTEB task. It enables us to directly use vector indexes while evaluating in MTEB.
    """

    def evaluate(self, model, split="test", score_function="cos_sim", **kwargs):
        try:
            from beir.retrieval.evaluation import EvaluateRetrieval
        except ImportError:
            raise Exception(
                "Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`"
            )

        if not self.data_loaded:
            self.load_data()

        corpus, queries, relevant_docs = (
            self.corpus[split],
            self.queries[split],
            self.relevant_docs[split],
        )
        # this is an artifact out of BEIR. https://github.com/beir-cellar/beir/blob/main/beir/retrieval/search/dense/faiss_search.py#L68
        # we run this here so that we don't make this assumption in our retriever, which could change.
        corpus_ids = sorted(
            corpus,
            key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
            reverse=True,
        )
        corpus = {cid: {"doc_id": cid, **corpus[cid]} for cid in corpus_ids}

        retriever = EvaluateRetrieval(
            model, score_function=score_function
        )  # or "cos_sim" or "dot"
        start_time = time()
        results, metrics = retriever.retrieve(corpus, queries)
        end_time = time()
        logger.info(
            "Time taken to retrieve: {:.2f} seconds".format(end_time - start_time)
        )

        ndcg, _map, recall, precision = retriever.evaluate(
            relevant_docs,
            results,
            retriever.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
        )
        mrr = retriever.evaluate_custom(
            relevant_docs, results, retriever.k_values, "mrr"
        )

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            **{
                "retrieval_latency_at_50": np.percentile(metrics.latencies, 50),
                "retrieval_latency_at_95": np.percentile(metrics.latencies, 95),
                "retrieval_latency_at_99": np.percentile(metrics.latencies, 99),
            },
        }

        return scores
