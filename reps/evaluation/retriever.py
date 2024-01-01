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
from python.sdk.indexes.indexing import Index
from python.sdk.processing.pipeline import ProcessingPipeline, Output
import logging

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
            processing_pipeline: ProcessingPipeline[Dict[str, str], Output],
            corpus_chunk_size: int = 250,

    ):
        super().__init__()
        self.index = index
        self.pipeline = processing_pipeline
        self.corpus_chunk_size = corpus_chunk_size

    def search(self,
               corpus: Dict[str,Dict[str, str]],
               queries: Dict[str, str],
               top_k: int,
               **kwargs) -> Dict[str, Dict[str, float]]:

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        query_ids = list(queries.keys())
        all_results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]

        query_embeddings = self.model.encode_queries(
            queries,
        )

        itr = range(0, len(corpus), self.corpus_chunk_size)

        if self.index.count() > 0:
            for batch_num, corpus_start_idx in enumerate(itr):
                logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
                corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

                embeddings = self.pipeline.process(corpus[corpus_start_idx:corpus_end_idx])
                self.index.add(embeddings)

        for query_itr in range(len(query_embeddings)):
            query = queries[query_itr]
            id = query_ids[query_itr]
            embedding = query_embeddings[query_itr]
            results = self.index.search(vector=embedding, limit=top_k)
            for result in results:
                all_results[id][result['id']] = result['score']

        return all_results