from typing import Dict, List
from math import pi, cos


params:Dict[str, List] = {
    "repeats": [50],
    "stagnation_limit": [200],
    "iterations_limit": [10000],
    "max_iterations": ["acc", 1000],
    "population_size": [10],
    "dimensions" : [20],

    "algorithm": [
        # Bat Algorithm
        # {
        #     "method" : "BA",
        #     "fmin" : 0, # minimal frequency
        #     "fmax" : 1, # maximal frequency
        #     "alpha": 0.9, # constant in (0, 1)
        #     "gamma": 5, # constant > 0

        #     "use_sigmoid": 1,
        #     "max_val" : 6,
        #     "min_val" : 0.8,
        #     "exponent" : 0.7,
        # },
        # Butterfly Optimization Algorithm
        {
            "method" : "BOA",
            "c" : 0.1, # sensory modality in [0, 1]
            "alpha" : 0.1, # power exponent in [0, 1]
            "prob": 0.9, # probability in [0, 1] to go near best solution
            
            "use_sigmoid": 1,
            "max_val" : 6,
            "min_val" : 0.8,
            "exponent" : 0.5,
        },
        {
            "method" : "BOA",
            "c" : 0.1, # sensory modality in [0, 1]
            "alpha" : 0.1, # power exponent in [0, 1]
            "prob": 0.9, # probability in [0, 1] to go near best solution
            
            "use_sigmoid": 1,
            "max_val" : 6,
            "min_val" : 0.8,
            "exponent" : 0.6,
        },
        {
            "method" : "BOA",
            "c" : 0.1, # sensory modality in [0, 1]
            "alpha" : 0.1, # power exponent in [0, 1]
            "prob": 0.9, # probability in [0, 1] to go near best solution

            "use_sigmoid": 1,
            "max_val" : 6,
            "min_val" : 0.8,
            "exponent" : 0.7,
        },
    ],

    "cost_function" : [
        {
            "equation" : {
                "Sphere" : 
                    lambda x_args: sum(map(
                        lambda x: x*x, 
                        x_args
                    )),
            },
            "min_x" : -100,
            "max_x" : 100,
            "accuracy": 0.0001,
            "min_y" : 0
        },
        {
            "equation" : {
                "Rastrigin" : 
                    lambda x_args: sum(map(
                        lambda x: x*x - 10*cos(2*pi*x) + 10, 
                        x_args
                    ))
            },
            "min_x" : -5.12,
            "max_x" : 5.12,
            "accuracy": 30,
            "min_y" : 0
        }
    ]
}
