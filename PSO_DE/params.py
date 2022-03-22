from typing import Dict, List
from math import pi, cos

params:Dict[str, List] = {
    "repeats": [10],
    "stagnation_limit": [200],
    "iterations_limit": [10000],
    "max_iterations": [200, "acc"],

    "population_size": [10, 100],
    "dimensions" : [20, 30],
    

    "algorithm": [
        # Differential Evolution
        {
            "method" : "PS",
            "inertia_weight" : 0.9, # w przedziale [0, 1]
            "cognitive_coef" : 1.9, # w przedziale [0, 2]
            "social_coef" : 0.1 # w przedziale [0, 2]
        },
        {
            "method" : "PS",
            "inertia_weight" : 0.9, # w przedziale [0, 1]
            "cognitive_coef" : 0.3, # w przedziale [0, 2]
            "social_coef" : 0.1 # w przedziale [0, 2]
        },
        {
            "method" : "PS",
            "inertia_weight" : 0.7, # w przedziale [0, 1]
            "cognitive_coef" : 0.7, # w przedziale [0, 2]
            "social_coef" : 0.7 # w przedziale [0, 2]
        },
        {
            "method" : "PS",
            "inertia_weight" : 0.1, # w przedziale [0, 1]
            "cognitive_coef" : 0.5, # w przedziale [0, 2]
            "social_coef" : 0.1 # w przedziale [0, 2]
        },
        {
            "method" : "DE",
            "scaling_factor" : 0.3, # F in [0, 1] for mutation
            "crossover_prob" : 0.1, # Pcr in [0, 1]
            "variant": "rand"
        },
        {
            "method" : "DE",
            "scaling_factor" : 0.3, # F in [0, 1] for mutation
            "crossover_prob" : 0.1, # Pcr in [0, 1]
            "variant": "best"
        },
        {
            "method" : "DE",
            "scaling_factor" : 0.5, # F in [0, 1] for mutation
            "crossover_prob" : 0.2, # Pcr in [0, 1]
            "variant": "best"
        }
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
