from typing import List


class SimStat():
    def __init__(self):
        self.stagnations_limit:int = 0
        self.iterations:List[int] = []
        self.best_adaptations:List[List[float]] = []
