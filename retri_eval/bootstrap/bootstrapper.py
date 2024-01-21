from collections import defaultdict
from typing import List, Union

import dspy
from dspy.teleprompt import (
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
    BootstrapFinetune,
)
from retri_eval.indexes.indexing import SearchResponse


class Bootstrapper:
    def __init__(self, llm=None, index=None, query_processor=None):
        self.llm = llm
        self.index = index
        self.query_processor = query_processor

        rm = RetrievalRM(index, query_processor)
        dspy.settings.configure(rm=rm, lm=llm)

    def bootstrap(self, samples=List[str]):
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.create_retrieval_match_metric(),
            max_bootstrapped_demos=2,
            num_candidate_programs=8,
            num_threads=12,
        )

        examples = [
            dspy.Example(document=sample.default_text()).with_inputs("document")
            for sample in samples
        ]
        teleprompter.compile(CoT(), trainset=examples)

    def create_retrieval_match_metric(self):
        def retrieval_match(example: dspy.Example, pred: dspy.Prediction):
            predicted_query = pred.answer

            processed_query = self.query_processor.process([predicted_query])[0]
            results = self.index.search(processed_query)

            doc = example.document
            return doc == results[0].text

        return retrieval_match


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("document -> query")

    def forward(self, document: str):
        return self.generate_answer(document=document)


class RetrievalRM(dspy.Retrieve):
    def __init__(self, index, query_processor, k: int = 3):
        self.index = index
        self.query_processor = query_processor
        super().__init__(k=k)

    def forward(self, query_or_queries=Union[str, List[str]]) -> dspy.Prediction:
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]

        all_query_results: List[List[SearchResponse]] = []
        for query in queries:
            _result = self.index.search(q=query, limit=self.k)
            all_query_results.append(_result)

        passages = defaultdict(float)

        for search_response in all_query_results:
            for result in search_response:
                passages[result.text] += result.score

        sorted_passages = sorted(passages.items(), key=lambda x: x[1], reverse=True)[
            : self.k
        ]
        return dspy.Prediction(passages=[passage for passage, _ in sorted_passages])
