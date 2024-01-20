from dataclasses import dataclass
from typing import List

@dataclass
class DataPair:
    """
    DataPair is essentially the query -> result pair. We maintain additional data about the DataPair, like ids.

    In the future, we might be concerned about format or language.
    """
    query: str
    doc: str
    doc_id: str
    is_positive: bool

@dataclass
class AggregatedQueryData:
    query: str
    positives: List[DataPair]
    negatives: List[DataPair]

