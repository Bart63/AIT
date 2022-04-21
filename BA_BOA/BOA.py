import numpy as np
import random
from typing import List
from sigmoid import generate_sigmoid

# Butterfly Optimization Algorithm

class BOA:
    def __init__(self, params) -> None:
        # Algo parameters
        self.population_size = params['population_size']
        self.c = params['c'] # sensory modality in [0, 1]
        self.alpha = params['alpha'] # power exponent in [0, 1]
        self.prob = params['prob'] # probability in [0, 1] to go near best solution

        # Cost function parameters
        self.dims = params['dimensions']
        self.x_range = params['x_range']
        self.cost_fun = params['cost_function']
        self.sigmoid = '' if not params['sigmoid'] else generate_sigmoid(
            params['sigmoid']['exponent'], 
            params['sigmoid']['max_val'], 
            params['sigmoid']['min_val']
        )
        self.sigmoid_counter = 0

        # Butterfly variables
        self.x = self.vector_init()
        self.F = []

        self.cost = [] # f(x)
        self.best_cost = "" # min(f(x))
        self.best_x = 0 # argmin(y(x)) 
        self.calc_cost()

    def vector_init(self, vectors_to_create=0, dims=0, x_range=0):
        if not vectors_to_create:
            vectors_to_create = self.population_size
        if not dims:
            dims = self.dims
        if not x_range:
            x_range = self.x_range

        vectors:List[List[float]] = []
        distance_range = x_range[1] - x_range[0]
        for i in range(vectors_to_create):
            vectors.append([])
            for _ in range(dims):
                rand_val =  random.uniform(0, 1) * distance_range + x_range[0]
                vectors[i].append(rand_val)
        return vectors
    
    def update_fragrance(self):
        self.sigmoid_counter += 1
        sigm_mul = 1 if not self.sigmoid else self.sigmoid(self.sigmoid_counter)
        self.F = [
            self.c * (self.cost_fun(pos) ** self.alpha) * sigm_mul
            for pos in self.x
        ]

    def update_pos(self):
        for i, x in enumerate(self.x):
            r = random.random()
            xi = np.array(x)
            if r < self.prob:
                g = np.array(self.best_x)
                self.x[i] = list(xi + (r*r * g - xi) * self.F[i])
            else:
                xj, xk = random.sample(self.x, 2)
                xj = np.array(xj)
                xk = np.array(xk)
                self.x[i] = list(xi + (r*r * xj - xk) * self.F[i])
            self.max_min_correct(self.x[i])

    def max_min_correct(self, x=0):
        for i, xi in enumerate(x):
            x[i] = min(max(xi, self.x_range[0]), self.x_range[1])
    
    def calc_cost(self, input_x=0):
        if not input_x:
            input_x = self.x
        self.cost = list(map(self.cost_fun, input_x))
        if isinstance(self.best_cost, str) or min(self.cost) < self.best_cost:
            self.best_cost = min(self.cost)
            self.best_x = input_x[self.cost.index(self.best_cost)]


def main():
    params = {
        'population_size': 10,
        'c': 0.1,
        'alpha': 0.1,
        'prob': 0.1,
        'dimensions': 4,
        'x_range' : (-100, 100),
        'cost_function': lambda x_args: sum(map(
                        lambda x: x*x, 
                        x_args
                    ))
    }
    BOA_algo = BOA(params)

    best_cost = BOA_algo.best_cost

    print(f'Start best cost: {best_cost}')

    for idx in range(1000):
        BOA_algo.update_fragrance()
        BOA_algo.update_pos()
        BOA_algo.calc_cost()
        if BOA_algo.best_cost < best_cost:
            best_cost = BOA_algo.best_cost
            print(f'New best cost in loop nr {idx+1}: {best_cost}')


if __name__ == '__main__':
    main()
