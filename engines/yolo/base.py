from abc import ABC, abstractmethod
import numpy as np

class YoloEngine(ABC):
    @abstractmethod
    def infer(self, tiles, metadata):
        pass

    @abstractmethod
    def get_perf(self):
        pass
