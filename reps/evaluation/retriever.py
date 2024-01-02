"""
Retriever is a wrapper for BEIR and MTEB to evaluate against our indexes.

MTEB relies on DenseRetrievalExactSearch, which as the name suggests, relies on a full exact search.
In most cases, this is fine. As the dataset grows, we might want to evaluate ANN approaches to
ensure our recall will be acceptable.

By wrapping our own index, we enable evaluation against indexes that can be used in production.

This line in MTEB could be more variable:
https://github.com/embeddings-benchmark/mteb/blob/02e84b2fa8d147a86b4896d8e57e83f36285f5c7/mteb/abstasks/AbsTaskRetrieval.py#L67
"""
from beir.retrieval.search import BaseSearch
from typing import Dict, List
from reps.indexes.indexing import Index
from reps.processing.pipeline import ProcessingPipeline, Output
import logging
from beir.retrieval.search.dense.util import cos_sim, dot_score

logger = logging.getLogger(__name__)


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
            query_processor: ProcessingPipeline[str, str],
            doc_processor: ProcessingPipeline[Dict[str, str], Output],
            corpus_chunk_size: int = 250,
    ):
        super().__init__()
        self.index = index
        self.query_processor = query_processor
        self.pipeline = doc_processor
        self.corpus_chunk_size = corpus_chunk_size

        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}

        self.results = {}

    def search(self,
               corpus: Dict[str,Dict[str, str]],
               queries: Dict[str, str],
               top_k: int,
               score_function: str,
               **kwargs) -> Dict[str, Dict[str, float]]:
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
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [{"doc_id": cid, **corpus[cid]} for cid in corpus_ids]

        query_ids = list(queries.keys())
        all_results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]

        query_embeddings = self.query_processor.process(
            queries,
        )

        itr = range(0, len(corpus), self.corpus_chunk_size)

        if self.index.count() == 0:
            for batch_num, corpus_start_idx in enumerate(itr):
                logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
                corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

                embeddings = self.pipeline.process(corpus[corpus_start_idx:corpus_end_idx])
                self.index.add(embeddings)

        for query_itr in range(len(query_embeddings)):
            query_id = query_ids[query_itr]
            embedding = query_embeddings[query_itr]
            results = self.index.search(vector=embedding, limit=top_k)
            for result in results:
                document_id = result.payload['doc_id']
                all_results[query_id][document_id] = result.score

        self.results = all_results
        return self.results


    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[float]:
        return self.query_processor.process(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        return self.pipeline.process(corpus, batch_size=batch_size, **kwargs)