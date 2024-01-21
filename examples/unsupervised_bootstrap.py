"""
This example shows how to bootstrap a dataset using unlabeled data.

We take the HotpotQA dataset, index it, and then bootstrap a dataset using 1,000 unlabeled samples from it.

The intuition is that we can generate relevant queries from the documents in the dataset, and then use those queries
to mine hard negatives from the index.

Each generated query should be validated against the index to make sure that the query returns the example document.
This code makes a hard assumption that the document will return as the top result, which may be too strict.
"""
import ir_datasets
from sentence_transformers import SentenceTransformer

from retri_eval.bootstrap.bootstrapper import Bootstrapper
from retri_eval.bootstrap.llm import DefaultLM
from retri_eval.indexes.manager import IndexAndProcessor, IndexManager
from retri_eval.indexes.usearch_index import USearchIndex
from retri_eval.processing.basic_query_processor import QueryProcessor
from retri_eval.processing.beir_title_processor import BeirTitleProcessor
import itertools

def main():
    dataset = ir_datasets.load("beir/hotpotqa/train")

    model_name = "BAAI/bge-small-en-v1.5"
    model = SentenceTransformer(model_name)

    index_name = "results/synthetic_queries/hotpot"
    index = USearchIndex(name=index_name, dims=384)
    doc_processor = BeirTitleProcessor(model, name=model_name)
    query_processor = QueryProcessor(model, name=model_name)

    if index.count() == 0:
        print("indexing dataset")
        manager = IndexManager([IndexAndProcessor(index, doc_processor)])
        manager.process_dataset(dataset, batch_size=32)


    dataset = ir_datasets.load("beir/hotpotqa/train")
    samples_for_bootstrapping = itertools.islice(dataset.docs_iter(), 1000)

    lm = DefaultLM()
    bootstrapper = Bootstrapper(lm, index, query_processor)

    bootstrapper.bootstrap(samples_for_bootstrapping)

if __name__ == "__main__":
    main()