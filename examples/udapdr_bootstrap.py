"""
This example shows how to bootstrap a dataset using unlabeled data.

We take the HotpotQA dataset, index it, and then bootstrap a dataset using 1,000 unlabeled samples from it.

The intuition is that we can generate relevant queries from the documents in the dataset, and then use those queries
to mine hard negatives from the index.

Each generated query should be validated against the index to make sure that the query returns the example document.
This code makes a hard assumption that the document will return as the top result, which may be too strict.
"""
import asyncio
from collections import defaultdict
import random
from typing import Dict, List
import dspy
import ir_datasets
from sentence_transformers import SentenceTransformer
import csv
# from retri_eval.bootstrap.synthetic_queries import SyntheticQueryGenerator
from retri_eval.bootstrap.llm import DefaultLM, LLMServer
from retri_eval.bootstrap.prompts import RetrievalTask, generate_udapdr_query
from retri_eval.bootstrap.synthetic_queries import SyntheticQueryGenerator
from retri_eval.indexes.manager import IndexAndProcessor, IndexManager
from retri_eval.indexes.usearch_index import USearchIndex
from retri_eval.processing.basic_query_processor import QueryProcessor
from retri_eval.processing.beir_title_processor import BeirTitleProcessor
import itertools
from tqdm import tqdm
import re


async def main():
    dataset = ir_datasets.load("beir/hotpotqa/train")
    lm = LLMServer(name="server", url="http://127.0.0.1:8888/v1/chat/completions")
    dspy.settings.configure(lm=lm, trace=[])

    query_generator = SyntheticQueryGenerator(llm=lm)
    bad_query_generator = SyntheticQueryGenerator(llm=lm, create_bad_queries=True)
    # this uses UDAPDR to generate good and bad queries.
    # Step 1: generate good and bad queries based on existing prompts.


    # Step 1/2: generate novel prompts based on the good and bad queries.
    # They will include random examples of documents that we have. This is the few-shot part.
    num_prompts = 4
    num_document_samples = 100
    ranker_count = 1
    num_examples_per_ranker = 3


    # we sample what rows and prompts we'll use in our few-shot stage. This way, we only generate the data we'll use
    # in stage 2 directly.
    rows_for_novel_prompts = random.sample(range(num_document_samples), num_examples_per_ranker * ranker_count)
    prompt_type_for_novel_prompts = [random.randrange(num_prompts) for _ in rows_for_novel_prompts]

    # map row to prompt type so that it's easy to look up the prompt we should use.
    row_to_prompt_type = dict(zip(rows_for_novel_prompts, prompt_type_for_novel_prompts))

    samples_for_bootstrapping = itertools.islice(dataset.docs_iter(), 100)
    novel_examples = []
    print("Step 1: generating prompts")
    with open(f"results/synthetic_queries/hotpot/generated_prompt_examples.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "query", "bad_query", "doc_text"])
        writer.writeheader()

        for ranker in range(ranker_count):
            current_prompt = ""
            for x in range(num_examples_per_ranker):
                row = rows_for_novel_prompts[ranker * num_examples_per_ranker + x]
                doc = next(itertools.islice(dataset.docs_iter(), row, None))

                query = query_generator.generate(doc.text)
                bad_query = bad_query_generator.generate(doc.text)
                bad_query = bad_query.replace("Bad Question", "").replace(": ", "")

                data = {"doc_id": doc.doc_id, "query": query, "doc_text": doc.text, 'bad_query': bad_query}
                writer.writerow(data)
                f.flush()

                current_prompt += f"Example {x}:\n"
                current_prompt += f"Document: {' '.join(doc.default_text().split(' ')[:256])}\n"
                current_prompt += f"Good Question: {query}\n"
                current_prompt += f"Bad Question: {bad_query}\n\n"


            novel_examples.append(current_prompt)

    novel_prompts = []
    # combine examples from novel_prompts into a single prompt per reranker.
    for i in range(0, len(novel_examples), num_examples_per_ranker):
        prompt = "\n".join(novel_examples[i:i + num_examples_per_ranker])
        novel_prompts.append(prompt)

    # Step 2: Generate synthetic queries for the sample set using our novel prompts. We are training n rerankers,
    # so we will iterate over the sample n times.
    print("Step 2: generating synthetic queries")
    query_doc_pairs = defaultdict(list) # list per ranker
    for i in range(ranker_count):
        with open(f"results/synthetic_queries/hotpot/synthetic_queries_{i}.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=["doc_id", "query", 'prompt'])

            samples_for_bootstrapping = itertools.islice(dataset.docs_iter(), 100)
            for doc in tqdm(samples_for_bootstrapping):
                prompt = novel_prompts[i]
                prompt += f"Example {num_examples_per_ranker}:\n"
                prompt += f"Document: {' '.join(doc.default_text().split(' ')[:256])}\n"

                response = lm.basic_request(prompt=prompt, n_predict=128)[0]
                query = response.replace("\n","").replace("\t","")
                query = query.replace("Good Question", "").replace(": ","")
                # quick hack to get only the query back. the LLM is also returning bad queries and instruction notes.
                query = query.split("?")[0] + '?'


                query_doc_pairs[i].append((query, doc.text))
                writer.writerow({'doc_id': doc.doc_id, 'query': query, 'prompt': prompt})
                f.flush()

    # Step 3: filter out queries that don't return the document in the top 20.
    # Additionally, for training rerankers, we collect triples of query, doc, label. label is either 1 or 0 if we find the doc
    # in the result set from the zero shot retrieval.
    # we determine negative docs based on what the zero shot ranker returns.
    print("Step 3: filtering out queries that don't return the document in the top 20")
    model_name = "BAAI/bge-small-en-v1.5"
    embedding_model = SentenceTransformer(model_name)

    index_name = "results/synthetic_queries/hotpot"
    index = USearchIndex(name=index_name, dims=384)
    doc_processor = BeirTitleProcessor(embedding_model, name=model_name)
    query_processor = QueryProcessor(embedding_model, name=model_name)

    if index.count() == 0:
        print("indexing dataset")
        manager = IndexManager([IndexAndProcessor(index, doc_processor)])
        samples_for_bootstrapping = itertools.islice(dataset.docs_iter(), 100)
        manager.process_dataset(samples_for_bootstrapping, batch_size=32)

    filtered_triples = defaultdict(list)
    for ranker_number, doc_pair_list in query_doc_pairs.items():
        num_filtered = 0
        with open(f"results/synthetic_queries/hotpot/training_triples_{ranker_number}_filtered.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=["query", "doc", "label", "doc_id"])
            for (query, doc) in doc_pair_list:
                processed_query = query_processor.process([query])[0]
                results = index.search(processed_query)

                collected_triples = []
                good_query = False
                for result in results[:20]:
                    # keep in line with what we're doing in the zero shot case.
                    processed_result = ' '.join(result.text.split(' ')[:256])

                    if processed_result == doc:
                        # found a positive match.
                        collected_triples.append((query, result.text, 1, result.doc_id))
                        good_query = True
                    else:
                        collected_triples.append((query, result.text, 0, result.doc_id))

                if good_query:
                    filtered_triples[ranker_number].extend(collected_triples)
                    writer.writerows({'query': query, 'doc': doc, 'label': label, 'doc_id': did} for (query, doc, label, did) in collected_triples)
                    f.flush()
                else:
                    print(f"query {query} did not return the document in the top 20 results")
                    num_filtered += 1

        print(f"filtered out {num_filtered} queries for ranker {ranker_number}")










if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
