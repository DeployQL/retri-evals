from abc import abstractmethod
from typing import List, NamedTuple
from itertools import islice

from tqdm import tqdm

from retri_eval.indexes.indexing import Index
from retri_eval.processing.pipeline import ProcessingPipeline


class Dataset:
    @abstractmethod
    def docs_iter(self):
        pass


class IndexAndProcessor(NamedTuple):
    index: Index
    processor: ProcessingPipeline


class IndexManager:
    """
    IndexManager handles iterating over a dataset and adding documents to a set of indexes.
    """

    def __init__(self, indexes_and_processors: List[IndexAndProcessor]):
        self.indexes = indexes_and_processors

    def process_dataset(self, dataset: Dataset, batch_size=100):
        for b in tqdm(batch(dataset.docs_iter(), batch_size)):
            for index_and_processor in self.indexes:
                index = index_and_processor.index
                doc_processor = index_and_processor.processor

                processed = doc_processor.process(b)
                index.add(processed)

        for index_and_processor in self.indexes:
            index = index_and_processor.index
            index.save()


def batch(it, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch
