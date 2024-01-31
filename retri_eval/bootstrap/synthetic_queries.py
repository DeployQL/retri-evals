from collections import defaultdict
from typing import List, Union

import dspy
from dspy.teleprompt import (
    BootstrapFewShotWithRandomSearch,
)
from dspy.primitives.assertions import (
    assert_transform_module,
    suggest_backtrack_handler,
)
from dspy.predict.retry import Retry
from retri_eval.bootstrap.prompts import CoT, RelevantQuery, NotRelevantQuery
from retri_eval.indexes.indexing import SearchResponse


class SyntheticQueryGenerator:
    def __init__(self, llm=None, create_bad_queries=False):
        self.llm = llm
        self.create_bad_queries = create_bad_queries

        dspy.settings.configure(trace=[])

        synthetic_query_generator = SynthesizeQueries(
            create_bad_queries
        ).map_named_predictors(Retry)
        self.generator = assert_transform_module(
            synthetic_query_generator, suggest_backtrack_handler
        )

    def generate(self, document):
        return self.generator(document).query


class SynthesizeQueries(dspy.Module):
    """
    SynthesizeQueries is a module that takes in a document and returns a query.
    """

    def __init__(self, create_bad_queries=False):
        self.cot = dspy.ChainOfThought(
            RelevantQuery if not create_bad_queries else NotRelevantQuery,
        )

    def forward(self, text: str) -> dspy.Prediction:
        context = []
        query = self.cot(context=context, document=text).query
        dspy.Suggest(
            len(query.split(" ")) > 3,
            "Query should be more than 3 words",
        )
        dspy.Suggest(
            query[-1] == "?",
            "Query should end with a question mark",
        )
        return dspy.Prediction(query=query)


class SynthesizeAndRetrieve(dspy.Module):
    """
    SynthesizeQueries is a module that takes in a document, creates a query, and then retrieves passages.
    """

    def __init__(self, index, query_processor):
        self.generate_queries = SynthesizeQueries()
        self.index = index
        self.query_processor = query_processor

    def forward(self, text: str) -> dspy.Prediction:
        query = self.cot(document=text).query
        processed_query = self.query_processor.process([query])[0]
        results = self.index.search(processed_query)
        return dspy.Prediction(
            query=query, passages=[result.text for result in results]
        )


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
