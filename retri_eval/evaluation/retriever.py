"""
Retriever is a wrapper for BEIR and MTEB to evaluate against our indexes.

Retrievers conform to the expected Model interface, so they can be passed to MTEB as models. But they also
provide a search() method that we use in our task to run the evaluation.
"""
from beir.retrieval.search import BaseSearch
from typing import Dict, List, Tuple, NamedTuple
from retri_eval.indexes.indexing import Index, SearchResponse
from retri_eval.processing.pipeline import ProcessingPipeline, Output
import logging
from beir.retrieval.search.dense.util import cos_sim, dot_score
import time

logger = logging.getLogger(__name__)


class RetrieverMetrics(NamedTuple):
    latencies: List[float]


class DenseRetriever(BaseSearch):
    """
    DeployQLRetriever evaluates retrievers based on BIER's BaseSearch class. This is how
    MTEB retrieves and scores.

    Unlike MTEB, we enable the index and processing to be external, which means the document embeddings
    can be reused.
    """

    def __init__(
        self,
        index: Index,
        query_processor: ProcessingPipeline[str, List[float]],
        doc_processor: ProcessingPipeline[Dict[str, str], Output],
        corpus_chunk_size: int = 250,
        queries_batch_size: int = 50,
    ):
        super().__init__()
        self.index = index
        self.query_processor = query_processor
        self.pipeline = doc_processor
        self.corpus_chunk_size = corpus_chunk_size
        self.queries_batch_size = queries_batch_size

        self.score_functions = {"cos_sim": cos_sim, "dot": dot_score}
        self.score_function_desc = {
            "cos_sim": "Cosine Similarity",
            "dot": "Dot Product",
        }

        self.results: Dict[str, Dict[str, float]] = {}

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        score_function: str,
        **kwargs
    ) -> Tuple[Dict[str, Dict[str, float]], List[float]]:
        """
        search() runs the entire evaluation process, returning a metrics object.

        Note there are a few scalability questions -- we keep pointers to the corpus in memory.
        We can offload this behavior to the index. All we need is a unique document id.

        :param corpus: The documents to index. Will only be indexed if the index is empty.
        :param queries: The queries to use in evaluation.
        :param top_k: how many documents to retrieve. this is used in metrics.
        :return: a mapping of metric name -> metric result.
        """
        if score_function not in self.score_functions:
            raise ValueError(
                "score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(
                    score_function
                )
            )

        # this is an artifact out of BEIR. https://github.com/beir-cellar/beir/blob/main/beir/retrieval/search/dense/faiss_search.py#L68
        corpus_ids = sorted(
            corpus,
            key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
            reverse=True,
        )
        corpus_list = [{"doc_id": cid, **corpus[cid]} for cid in corpus_ids]

        query_ids = list(queries.keys())
        all_results: Dict[str, Dict[str, float]] = {qid: {} for qid in query_ids}
        queries_list = [queries[qid] for qid in queries]

        query_embeddings = self.encode_queries(
            queries_list,
            batch_size=self.queries_batch_size,
        )

        itr = range(0, len(corpus_list), self.corpus_chunk_size)

        if self.index.count() == 0:
            for batch_num, corpus_start_idx in enumerate(itr):
                logger.info("Encoding Batch {}/{}...".format(batch_num + 1, len(itr)))
                corpus_end_idx = min(
                    corpus_start_idx + self.corpus_chunk_size, len(corpus_list)
                )

                embeddings = self.encode_corpus(
                    corpus_list[corpus_start_idx:corpus_end_idx], batch_size=50000
                )
                self.index.add(embeddings)
        else:
            logger.info("Skipping indexing. Given Index is not empty.")

        latencies = []
        for query_itr in range(len(query_embeddings)):
            query_id = query_ids[query_itr]
            embedding = query_embeddings[query_itr]

            start = time.perf_counter()
            results: List[SearchResponse] = self.index.search(
                vector=embedding, limit=top_k
            )
            duration = time.perf_counter() - start
            latencies.append(duration)
            for result in results:
                document_id = result.doc_id
                all_results[query_id][document_id] = result.score

        return all_results, RetrieverMetrics(latencies=latencies)

    def encode_queries(
        self, queries: List[str], batch_size: int, **kwargs
    ) -> List[List[float]]:
        return self.query_processor.process(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        return self.pipeline.process(corpus, batch_size=batch_size, **kwargs)
