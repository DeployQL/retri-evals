from typing import List

import numpy as np

from retri_eval.processing.pipeline import ProcessingPipeline


class QueryProcessor(ProcessingPipeline[str, List[np.ndarray]]):
    def __init__(self, model, name="", version=""):
        super().__init__(name, version)
        self.model = model

    def process(
        self, batch: List[str], batch_size: int = 0, **kwargs
    ) -> List[np.ndarray]:
        return self.model.encode_queries(batch)
