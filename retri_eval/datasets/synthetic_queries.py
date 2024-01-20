from retri_eval.datasets.types import DataPair
from retri_eval.indexes.indexing import Index
from retri_eval.processing.pipeline import ProcessingPipeline
from typing import List
from llama_cpp import Llama


class SyntheticQueryMiner:
    def __init__(self, llm: Llama, index: Index, query_processor: ProcessingPipeline):
        self.LLM = llm
        self.index = index
        self.query_processor = query_processor

    def create_synthetic_query(self, unlabeled_doc: str, examples: List[DataPair]):
        prompt = self._create_prompt(unlabeled_doc, examples)
        generated = self.LLM(prompt)["choices"][0]['text']
        return generated


    def _create_prompt(self, unlabeled_doc: str, examples: List[DataPair]) -> str:
        if examples:
            return self._create_prompt_with_examplars(unlabeled_doc, examples)
        else:
            return self._create_unsupervised_prompt(unlabeled_doc)

    def _create_unsupervised_prompt(self, unlabeled_doc: str) -> str:
        return f'''
Create one question using only the context provided starting with "What", "How" or "Why". Only respond with the question, don't say anything else (unecessary starting words, hints, etc.)
<document>
{unlabeled_doc}
</document>
'''

    def _create_prompt_with_examplars(self, unlabeled_doc: str, examples: List[DataPair]) -> str:
        templates = []
        for i, example in enumerate(examples):
            templates += [
                f'''

Example {i+1}:
document: {example.doc}
query: ${example.query}
                '''
            ]
        string_templates = '\n'.join(templates)

        return f'''
These are examples of queries with sample relevant documents for
each query. Create one question using only the context provided starting with "What", "How" or "Why". Only respond with the question, don't say anything else (unecessary starting words, hints, etc.)

{string_templates}

Example {len(templates)}:
document: {unlabeled_doc}
query:
            '''


class QueryEvaluator:
    def __init__(self, index: Index, query_processor: ProcessingPipeline):
        self.index = index
        self.query_processor = query_processor


    def is_valid(self, query: str, expected_doc: str, doc_id: int | None = None) -> bool:
        """
        When we generate a query, we want to search the index and make sure that the document used to generate the query
        shows up first. This gives us confidence that the query is good for training data. We don't want queries that are
        too generic.
        :param query:
        :return: bool. True if valid.
        """
        embedding = self.query_processor.process([query])[0]
        results = self.index.search(embedding, 1)
        if doc_id is None and results[0].text != expected_doc:
            print(f"Query: {query} is not valid. Expected: {expected_doc}, got: {results[0].text}. score: {results[0].score}")
            return False
        elif doc_id is not None and results[0].doc_id != doc_id:
            print(f"Query: {query} is not valid. Expected: {expected_doc}, got: {results[0].text}. score: {results[0].score}")
            return False
        else:
            return True