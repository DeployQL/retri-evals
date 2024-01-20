from typing import Dict, List, Tuple
from retri_eval.indexes.indexing import Index
from retri_eval.processing.pipeline import ProcessingPipeline, Output
from retri_eval.datasets.types import AggregatedQueryData


class HardNegativeMiner:
    """
    HardNegativeMiner uses a prebuilt Index to search for similar results to the query.
    """
    def __init__(
            self,
            index: Index,
            query_processor: ProcessingPipeline[str, List[float]],
            doc_processor: ProcessingPipeline[Dict[str, str], Output],
    ):
        self.index = index
        self.query_processor = query_processor
        self.pipeline = doc_processor

    def mine_single(self, data: AggregatedQueryData, num_negatives=5) -> List[str]:
        """
        mine a single triplet. This leverages our pipelines and index to return candidate negatives, filtering out
        any that are listed as positives.
        :param data: TrainingTiplet
        :return:  a list of text that can be used in training.
        """
        doc = self.query_processor.process(data.query)
        results = self.index.search(doc.embedding)

        filtered_by_id = [x for x in results if x.doc_id not in [y.doc_id for y in data.positives]]
        filtered_by_text = [x for x in filtered_by_id if x.doc not in [y.doc for y in data.positives]]

        return [x.text for x in filtered_by_text][:num_negatives]