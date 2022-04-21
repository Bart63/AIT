import numpy as np
import random
from math import exp
from typing import List
from sigmoid import generate_sigmoid

# Bat Algorithm

class BA:
    def __init__(self, params) -> None:
        # Algo parameters
        self.population_size = params['population_size']
        self.f_range = params['f_range'] # (f_min, f_max)
        self.alpha = params['alpha'] # constant in (0, 1)
        self.gamma = params['gamma'] # constant > 0

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

        self.cost = [] # f(x)
        self.best_cost = "" # min(f(x))
        self.best_x = 0 # argmin(y(x)) 

        self.variables_init()

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

    def variables_init(self):
        # Population variables
        self.x = self.vector_init()
        self.v = self.vector_init(x_range=(0, 0))

        self.f = self.vector_init(dims=1, x_range=(0,0))
        self.update_freq()

        # First calculation of cost
        # Best solution selection
        self.calc_cost()

        # Loudness, emission rate
        self.Ai = self.vector_init(dims=1, x_range=(1, 2))
        self.r0 = self.vector_init(dims=1, x_range=(0, 1))
        self.ri = [
            [r[0]]
            for r in self.r0
        ]

        # Iteration for changing emission rate
        self.iteration_of_r = 1
    
    def update_pos_vel(self):
        x_best = np.array(self.best_x)

        for i, vi in enumerate(self.v):
            xi = np.array(self.x[i])
            vi = np.array(vi)
            fi = np.array(self.f[i])
            self.v[i] = list(vi + (x_best - xi) * fi) # Corrected

            vi = np.array(self.v[i])
            self.x[i] = list(xi + vi)

            r1 = random.random()
            if r1 < self.ri[i][0]:
                epsilon = random.uniform(-1, 1)

                mean_A = sum([
                    a[0] for a in self.Ai
                ])/len(self.Ai)
                self.x[i] = list(x_best + epsilon*mean_A)
            self.max_min_correct(self.x[i])
                    
    def update_A_r(self):
        self.iteration_of_r += 1
        for i, Ai in enumerate(self.Ai):
            r2 = random.random()
            if r2 < Ai[0] and self.cost_fun(self.x[i]) < self.best_cost:
                self.Ai[i][0] = self.alpha * Ai[0]
                self.ri[i][0] = self.r0[i][0] * (1 - exp(-self.gamma*self.iteration_of_r))

    def update_freq(self):
        self.sigmoid_counter += 1
        sigm_mul = 1 if not self.sigmoid else self.sigmoid(self.sigmoid_counter)
        f_min = self.f_range[0]
        f_max = self.f_range[1]

        for i, _ in enumerate(self.f):
            beta = self.vector_init(vectors_to_create=1, x_range=(0, 1))[0]

            self.f[i] = [
                (f_min + (f_max - f_min) * b) * sigm_mul
                for b in beta
            ]

    def max_min_correct(self, x=0):
        if not x:
            x = self.x
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
        'f_range': (0, 2),
        'alpha': 0.1,
        'gamma': 1,
        'dimensions': 4,
        'x_range' : (-100, 100),
        'cost_function': lambda x_args: sum(map(
                        lambda x: x*x, 
                        x_args
                    ))
    }
    BA_algo = BA(params)

    best_cost = BA_algo.best_cost

    print(f'Start best cost: {best_cost}')

    for idx in range(1000):
        BA_algo.update_freq()
        BA_algo.update_pos_vel()
        BA_algo.update_A_r()
        BA_algo.calc_cost()
        #print(BA_algo.f)
        if BA_algo.best_cost < best_cost:
            best_cost = BA_algo.best_cost
            print(f'New best cost in loop nr {idx+1}: {best_cost}')


if __name__ == '__main__':
    main()
