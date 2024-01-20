from typing import Dict, List

from retri_eval.datasets.types import AggregatedQueryData, DataPair
from retri_eval.indexes.indexing import Index
from retri_eval.processing.pipeline import ProcessingPipeline, Output
from retri_eval.datasets.hard_negative_miner import HardNegativeMiner
from retri_eval.datasets.synthetic_queries import SyntheticQueryMiner, QueryEvaluator
from retri_eval.datasets.llm import DefaultQueryLLM

class TrainingDataManager:
    """
    TrainingDataManager exists to facilitate training a ranking model.

    There are a few scenarios we should prepare for:
    1. The data is unlabeled, and we need to mine negatives and create synthetic positives.
    2. The data is labeled, and we can expand negatives or create fully synthetic data to augment it.
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
        self.negative_miner = HardNegativeMiner(
            index,
            query_processor,
            doc_processor,
        )
        self.synthetic_query_miner = SyntheticQueryMiner(llm=DefaultQueryLLM, index=index, query_processor=query_processor)
        self.query_evalutor = QueryEvaluator(index, query_processor)

    def process_pairs(self, data: List[DataPair], num_negatives=5, num_examples=1):
        """
        process_pairs is responsible for convert pairs of data to usable training data. This may mean mining hard negatives
        or synthesizing queries for the given data.
        :param data:
        :return:
        """
        data_with_positives = []
        unlabeled_data = []

        aggregate_samples = self._aggregate_pairs(data)
        for aggregate in aggregate_samples.values():
            if not aggregate.positives:
                unlabeled_data.append(aggregate)
            else:
                data_with_positives.append(aggregate)

        training_pairs = []
        # we have data with positives, where we can grab some negative samples pretty easily.
        # we also have unlabeled data, where we will create synthetic queries.
        for labeled in data_with_positives:
            if len(labeled.negatives) < num_negatives:
                negatives = self.negative_miner.mine_single(labeled, num_negatives=num_negatives-len(labeled.negatives))
                existing_negatives = [x.doc for x in labeled.negatives]
                for neg in existing_negatives + negatives:
                    training_pairs.append((labeled.query, neg))
            else:
                existing_negatives = [x.doc for x in labeled.negatives]
                for neg in existing_negatives:
                    training_pairs.append((labeled.query, neg))


        # we need exemplary (query, doc) pairs to use as context. for now, we grab the top training pairs.
        exemplars = [DataPair(query=query, doc=doc, doc_id='', is_positive=False) for query, doc in training_pairs[:num_examples]]

        seen = set()
        rejected = set()
        for unlabeled in unlabeled_data:
            for neg in unlabeled.negatives:
                if neg.doc in seen:
                    continue
                synthesized_query = self.synthetic_query_miner.create_synthetic_query(neg.doc, exemplars)
                is_valid = self.query_evalutor.is_valid(synthesized_query, neg.doc, neg.doc_id)
                if not is_valid:
                    rejected.add(neg.doc)
                    continue
                seen.add(neg.doc)
                training_pairs.append((synthesized_query, neg.doc))

        return training_pairs


    def _aggregate_pairs(self, data: List[DataPair]) -> Dict[str, AggregatedQueryData]:
        results: Dict[str, AggregatedQueryData] = {}
        for pair in data:
            if pair.query not in results:
                results[pair.query] = AggregatedQueryData(pair.query, negatives=[], positives=[])

            if pair.is_positive:
                results[pair.query].positives.append(pair)
            else:
                results[pair.query].negatives.append(pair)

        return results

