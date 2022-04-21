from itertools import product
from typing import List, Dict

import get_params as gp
import params as parameters
from simstat import SimStat
from save_stats import save_stats

from BA import  BA
from BOA import BOA


def simulate_BA(params):
    sim_stats = SimStat()

    iterations_limit = params['iterations_limit']
    stagnation_limit = params['stagnation_limit']

    max_iterations = params['max_iterations']
    acc = params['cost_function']['accuracy']
    min_y = params['cost_function']['min_y']

    BA_params = gp.get_params_BA(params)

    for _ in range(params["repeats"]):
        best_adaptations:List[float] = []

        # Start BA
        bats = BA(BA_params)
        best_adaptations.append(bats.best_cost)

        stagnations = 0
        iterations = 0
        done = False
        while not done:
            bats.update_freq()
            bats.update_pos_vel()
            bats.update_A_r()
            bats.calc_cost()
            # End BA

            best_adaptations.append(bats.best_cost)

            if best_adaptations[-2] == bats.best_cost:
                stagnations += 1
                if stagnations == stagnation_limit:
                    sim_stats.stagnations_limit += 1
                    break
            else:
                stagnations = 0

            iterations += 1
            if isinstance(max_iterations, int):
                done = iterations >= max_iterations 
            else:
                done = (bats.best_cost - min_y) < acc

            if iterations > iterations_limit:
                sim_stats.iteratioons_limit += 1
                break
        sim_stats.best_adaptations.append(best_adaptations)
        sim_stats.iterations.append(iterations)
    save_stats(params, sim_stats)


def simulate_BOA(params):
    sim_stats = SimStat()

    iterations_limit = params['iterations_limit']
    stagnation_limit = params['stagnation_limit']

    max_iterations = params['max_iterations']
    acc = params['cost_function']['accuracy']
    min_y = params['cost_function']['min_y']

    BOA_params = gp.get_params_BOA(params)

    for _ in range(params["repeats"]):
        best_adaptations:List[float] = []

        # Start BOA
        butterflies = BOA(BOA_params)
        best_adaptations.append(butterflies.best_cost)

        stagnations = 0
        iterations = 0
        done = False
        while not done:
            butterflies.update_fragrance()
            butterflies.update_pos()
            butterflies.calc_cost()
            # End BOA

            best_adaptations.append(butterflies.best_cost)

            if best_adaptations[-2] == butterflies.best_cost:
                stagnations += 1
                if stagnations == stagnation_limit:
                    sim_stats.stagnations_limit += 1
                    break
            else:
                stagnations = 0

            iterations += 1
            if iterations>2000:
                return
            if isinstance(max_iterations, int):
                done = iterations >= max_iterations 
            else:
                done = abs(butterflies.best_cost - min_y) < acc
            
            if iterations > iterations_limit:
                sim_stats.iteratioons_limit += 1
                break
        sim_stats.best_adaptations.append(best_adaptations)
        sim_stats.iterations.append(iterations)
    save_stats(params, sim_stats)


def main():
    print('Start program')
    params_dict:Dict[str, List] = parameters.params
    key_names:List[str] = list(params_dict.keys())
    all_combinations_params:List = list(product(*list(params_dict.values())))
    
    combinations_param_list:Dict = [
        {k:v for k, v in zip(key_names, param)}
        for param in all_combinations_params
    ]

    for i, params in enumerate(combinations_param_list):
        print(f'Simulation number: {i+1}')
        
        if params['algorithm']['method'] == 'BA':
            simulate_BA(params)
        elif params['algorithm']['method'] == 'BOA':
            simulate_BOA(params)


if __name__ == '__main__':
    main()
