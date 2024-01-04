# Retrieval Evaluation Pipelines <sup>alpha</sup>
### RAG evaluation framework for faster iteration


## About retri-eval
Evaluating all of the components of a RAG pipeline is challenging. We didn't find a
great existing solution that was
1. flexible enough to fit on top of our document and query processing.
2. gave us confidence in scaling the database up without increasing latency or costs.
3. encouraged reuse of components.

retri-eval aims to be unopinionated enough that you can reuse any existing pipelines you have.

## Built With
- MTEB
- BEIR
- Pydantic

## Getting Started
### Installation
```bash
pip install retri-eval
```
### Define your data type
We use Pydantic to make sure that the index receives the expected data.

To use MTEB and BEIR datasets, retri-eval expects your data to provide a `doc_id` field.
This is set inside of our retriever and is how BEIR evaluates your results.

Below, we create a `QdrantDocument` that specifically indexes text alongside the embedding.
```python
class QdrantDocument(MTEBDocument):
    id: str
    doc_id: str
    embedding: List[float]
    text: str
```

### Create a Document Processing Pipeline
A document processor encapsulates the logic to translate from raw data to our defined type.

```python
class DocumentProcessor(ProcessingPipeline[Dict[str, str], QdrantDocument]):
    def __init__(self, model, name='', version=''):
        super().__init__(name, version)
        self.model = model

    def process(self, batch: List[Dict[str, str]], batch_size: int=0, **kwargs) -> List[QdrantDocument]:
        chunker = lambda x: [x]

        results = []
        for x in batch:
            doc = MTEBDocument(**x)

            chunks = chunker(doc.text)
            embedding = self.model.encode(chunks)
            for i, chunk in enumerate(chunks):
                results.append(QdrantDocument(
                    id=uuid.uuid4().hex,
                    doc_id=doc.doc_id,
                    text=chunk,
                    embedding=embedding[i],
                ))
        return results
```

### Create a Query Processing Pipeline
Similar to document processing, we need a way to convert strings to something the index will understand.

For dense retrieval, we return embeddings from a model.

```python
class QueryProcessor(ProcessingPipeline[str, List[float]]):
    def __init__(self, model, name = '', version = ''):
        super().__init__(name, version)
        self.model = model

    def process(self, batch: List[str], batch_size: int=0, **kwargs) -> List[List[float]]:
        return self.model.encode_queries(batch)
```

### Define a Retriever
The Retriever class acts as our interface to processing. It defines our search behavior
over the index. retri-eval defines a DenseRetriever for MTEB.

```python
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
```

### Use our MTEB Tasks
MTEB makes it difficult to use our own search functionality, so we wrote our own MTEB Task
and extended MTEB tasks to use it.

This lets us bring our own indexes and define custom searching behavior. We're hoping to upstream this in the future.

````python
from retri-eval.evaluation.mteb_tasks import CQADupstackEnglishRetrieval

eval = MTEB(tasks=[CQADupstackEnglishRetrieval()])
results = eval.run(retriever, verbosity=2, overwrite_results=True, output_folder=f"results/{id}")

print(json.dumps(results, indent=1))
````
results:
```bash
{
 "CQADupstackEnglishRetrieval": {
  "mteb_version": "1.1.1",
  "dataset_revision": null,
  "mteb_dataset_name": "CQADupstackEnglishRetrieval",
  "test": {
   "ndcg_at_1": 0.37006,
   "ndcg_at_3": 0.39158,
   "ndcg_at_5": 0.4085,
   "ndcg_at_10": 0.42312,
   "ndcg_at_100": 0.46351,
   "ndcg_at_1000": 0.48629,
   "map_at_1": 0.29171,
   "map_at_3": 0.35044,
   "map_at_5": 0.36476,
   "map_at_10": 0.3735,
   "map_at_100": 0.38446,
   "map_at_1000": 0.38571,
   "recall_at_1": 0.29171,
   "recall_at_3": 0.40163,
   "recall_at_5": 0.44919,
   "recall_at_10": 0.49723,
   "recall_at_100": 0.67031,
   "recall_at_1000": 0.81938,
   "precision_at_1": 0.37006,
   "precision_at_3": 0.18535,
   "precision_at_5": 0.13121,
   "precision_at_10": 0.07694,
   "precision_at_100": 0.01252,
   "precision_at_1000": 0.00173,
   "mrr_at_1": 0.37006,
   "mrr_at_3": 0.41943,
   "mrr_at_5": 0.4314,
   "mrr_at_10": 0.43838,
   "mrr_at_100": 0.44447,
   "mrr_at_1000": 0.44497,
   "retrieval_latency_at_50": 0.07202814750780817,
   "retrieval_latency_at_95": 0.09553944145009152,
   "retrieval_latency_at_99": 0.20645513817435127,
   "evaluation_time": 538.25
  }
 }
}
```

## Roadmap
retri-eval is still in active development. We're planning to add the following functionality:

- [ ] Support reranking models
- [ ] Add support for hybrid retrieval baselines
- [ ] Support for automatic dataset generation
- [ ] Support parallel execution
- [ ] Add support for latency and cost benchmarks

# What dataset to evaluate on
retri-eval is currently integrated into MTEB for retrieval tasks only, but we're working on more.

[MTEB's available tasks](https://github.com/embeddings-benchmark/mteb/tree/main?tab=readme-ov-file#available-tasks)

We also recommend building your own internal dataset, but this can be time consuming and potentially
error prone. We'd love to chat if you're working on this.

## License
Distributed under the AGPL-3 License. If you need an alternate license, please reach out.


# Let's Chat!
Reach out! Our team has experience working on petabyte-scale search and analytics applications.
We'd love to hear what you're working on and see how we can help.

Matt - matt _[at]_ deployql.com - [Or Schedule some time to chat on my calendar](https://calendar.app.google/obJmewkwVSuUcSK1A)



## Acknowledgements
- [MTEB](https://github.com/embeddings-benchmark/mteb)