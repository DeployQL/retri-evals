# Retrieval Evaluation Pipelines <sup>alpha</sup>
### RAG evaluation framework for faster iteration

REPS contains abstractions that we've found helpful for benchmarking different retrieval pipelines.

Built on MTEB, REPS helps bridge the gap between benchmarking and production.

## Why REPS
RAG has been a huge focus lately for Gen AI, and MTEB provides open source, academic datasets
for evaluating retrieval. This can be a great way to quickly evaluate retrieval without labeling your own dataset.

REPS helps you benchmark different parts of retrieval against your dataset:
1. Text cleaning
2. Chunking
3. Embedding Model
4. Query expansions
4. Retrieval algorithm (sparse vs dense vs hybrid)
5. Reranking algorithm
6. Number of results to return

### Extending beyond relevancy
Relevancy was only one part of our benchmarking.
We wanted to know what latency to expect and whether our database will scale. REPS
aims to add additional functionality to benchmark latency, cost, and database updates.

REPS helps you benchmark faster. You can reuse a populated index while iterating on
query processing and reranking. How data is processed is left up to you, and REPS integrates
it into evaluation.

## REPS and MTEB
We love MTEB and thank their maintainers for their work. As this repo finds it's footing,
we want to upstream work that makes sense to be in MTEB.

We see the future of REPS as moving toward:
- evaluations of tasks on top of retrieval
- debugging and introspection of retrieval pipelines
- Facilitating internal dataset creation.


# Roadmap
REPS is still in beta. We're planning to add the following functionality:

- [ ] Support for automatic dataset generation
- [ ] Support parallel execution
- [ ] Add support for hybrid retrieval baselines
- [ ] Add support for latency and cost benchmarks

# Usage
REPS exposes four fundamental interfaces for you to build upon.
1. Index
2. Query Processor
3. Document Processor
4. Retriever


REPS tries to be unopinionated about your processing requirements. The below
code snippet shows how to define your own processors while leveraging REPS' DenseRetriever
and MTEB tasks that we've extended.

We expect that Processors will be named and versioned, and REPS outputs
results based on these processors.

```python
from reps.evaluation.mteb_tasks import CQADupstackEnglishRetrieval
from reps.evaluation.retriever import DenseRetriever
from reps.indexes.qdrant_index import QdrantIndex, QdrantDocument
from reps.indexes.indexing import MTEBDocument
from reps.processing.pipeline import ProcessingPipeline, Input, Output
from FlagEmbedding import FlagModel
from qdrant_client.models import VectorParams, Distance
from mteb import MTEB

model_name ="BAAI/bge-small-en-v1.5"
model = FlagModel(model_name,
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                  use_fp16=True)

index = QdrantIndex("CQADupstackEnglish", vector_config=VectorParams(size=384, distance=Distance.COSINE))
doc_processor = DocumentProcessor(model, name=model_name)
query_processor = QueryProcessor(model, name=model_name)

retriever = DenseRetriever(
    index=index,
    query_processor=query_processor,
    doc_processor=doc_processor,
)

eval = MTEB(tasks=[CQADupstackEnglishRetrieval()])
results = eval.run(retriever, verbosity=2, overwrite_results=True, output_folder=f"results/{id}")
```

# What dataset to evaluate on
Building your own internal evaluation dataset is going to be the highest signal while
also being the most time consuming.

REPS is currently integrated into MTEB for retrieval tasks only, but we're working on more.

[MTEB's available tasks](https://github.com/embeddings-benchmark/mteb/tree/main?tab=readme-ov-file#available-tasks)


# Have questions?
Reach out! Our team has experience working on petabyte-scale search and analytics applications.
We'd love to hear what you're working on.
---
## Notable Repositories
[MTEB](https://github.com/embeddings-benchmark/mteb)