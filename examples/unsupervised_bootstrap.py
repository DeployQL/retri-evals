"""
This example shows how to bootstrap a dataset using unlabeled data.

We sample from the HotpotQA dataset to generate unseen queries.
"""
import asyncio

import ir_datasets

from retri_eval.bootstrap.llm import LLMServer

from retri_eval.bootstrap.synthetic_queries import SyntheticQueryGenerator
import itertools
import dspy


async def main():
    dataset = ir_datasets.load("beir/hotpotqa/train")

    dataset = ir_datasets.load("beir/hotpotqa/train")
    samples_for_bootstrapping = itertools.islice(dataset.docs_iter(), 10)

    lm = LLMServer(name="server", url="http://127.0.0.1:8888/v1/chat/completions")
    dspy.settings.configure(lm=lm, trace=[])


    generator = SyntheticQueryGenerator(llm=lm)

    for i, doc in enumerate(samples_for_bootstrapping):
        query = generator.generate(doc.text)
        print(query)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
