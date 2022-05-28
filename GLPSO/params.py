from typing import Dict, List
from math import pi, cos
import cost_functions as cs
import numpy as np


params:Dict[str, List] = {
    "repeats": [50],
    "stagnation_limit": [500],
    "max_iterations": [200], # 1000

    "population_size": [30], # 10
    "dimensions" : [80], # 30 ~ 100
    

    "algorithm": [
        {
            "mutation_prob" : 0.01, # w przedziale [0, 1]
            "inertia_weight" : 0.7298, # w przedziale [0, 2]
            "amplification_factor": 0.5, # F w przedziale [0, 1]
            "c" : 0.149618, # w przedziale [0, 2]
            "c1" : 0.149618, # w przedziale [0, 2]
            "c2" : 0.149618, # w przedziale [0, 2]
            "sg" : 7
        }
    ],

    "cost_function" : [
        {
            "equation" : {
                "Sphere" : 
                    cs.sphere,
            },
            "min_x" : -100,
            "max_x" : 100
        },
        {
            "equation" : {
                "Ackley" : 
                    cs.ackley,
            },
            "min_x" : -32,
            "max_x" : 32
        },
        {
            "equation" : {
                "Rosenbrock" : 
                    cs.rosenbrock,
            },
            "min_x" : -2048,
            "max_x" : 2048
        },
        {
            "equation" : {
                "SumOfDifferentPower" : 
                    cs.sum_different_power,
            },
            "min_x" : -1,
            "max_x" : 1
        },
        {
            "equation" : {
                "Griewank" : 
                    cs.griewank,
            },
            "min_x" : -600,
            "max_x" : 600
        },
        {
            "equation" : {
                "Rastrigin" : 
                    cs.rastrigin,
            },
            "min_x" : -5.12,
            "max_x" : 5.12
        },
        {
            "equation" : {
                "Zakharov" : 
                    cs.zakharov,
            },
            "min_x" : -10,
            "max_x" : 10
        }
    ]
}
