from glob import glob
import matplotlib.pyplot as plt
from swarm import Swarm
import params as parameters
from simstat import SimStat
from typing import List, Dict, Tuple
from itertools import product
from DE import DE
import statistics
import shutil
import glob
import json
import os


OUTPUT_FOLDER = "output"


def save_stats(params, sim_stats:SimStat):
    equation_name = list(params['cost_function']['equation'].keys())[0]
    exp_name = f"{equation_name}_{params['algorithm']['method']}_E{params['max_iterations']}"
    exp_name += f"_P{params['population_size']}_D{params['dimensions']}"

    algo_dict = params['algorithm']
    key_names:List[str] = list(algo_dict.keys())
    all_combinations_params:List = list(algo_dict.values())

    results_dict = {
        k:v
        for k, v in zip(key_names, all_combinations_params)
    }

    for k, v in results_dict.items():
        if k == 'method':
            continue
        exp_name += f'_{k[:1]}{v}'
    
    exp_name = exp_name.replace('.', '')

    directory_path = os.path.join(OUTPUT_FOLDER, exp_name)
    if os.path.exists(directory_path):
        folders_count = len(glob.glob(directory_path))
        directory_path += '_'+str(folders_count)
    os.mkdir(directory_path)

    mean_iterations = sum(sim_stats.iterations)/len(sim_stats.iterations)

    solutions = [
        adapts[-1]
        for adapts in sim_stats.best_adaptations
    ]

    mean_solution = sum(solutions)/len(solutions)
    sd_solution = statistics.stdev(solutions)

    stats_dict ={
        "equation_name": equation_name,

        "max_iterations": params['max_iterations'],
        "population_size": params['population_size'],
        "dimensions": params['dimensions'],
        
        "repeats": params['repeats'],
        "stagnation_limit": params['stagnation_limit'],
        "iterations_limit": params['iterations_limit'],

        "min_x": params['cost_function']['min_x'],
        "max_x": params['cost_function']['max_x'],
        "min_y": params['cost_function']['min_y'],
        "accuracy": params['cost_function']['accuracy'],

        "stagnation_limit": sim_stats.stagnations_limit,
        "iteratioons_limit": sim_stats.iteratioons_limit,

        "mean_iterations": mean_iterations,
        "mean_solution": mean_solution,
        "sd_solution": sd_solution
    }

    results_dict.update(stats_dict)

    json_path = os.path.join(directory_path, 'stats.json')
    with open(json_path, 'w') as out_stats:
        json.dump(results_dict, out_stats)

    plot_folder_path = os.path.join(directory_path, 'graphs')
    os.mkdir(plot_folder_path)

    for repeat, adaptations in enumerate(sim_stats.best_adaptations):
        plt.plot(adaptations)
        plt.title("Najlepsza adaptacja w kolejnych iteracjach")
        plt.xlabel("Iteracje")
        plt.ylabel("Adaptacja")
        plot_path = os.path.join(plot_folder_path, f'{repeat}.png')
        plt.savefig(plot_path)
        plt.clf()

    for repeat, adaptations in enumerate(sim_stats.best_adaptations):
        plt.plot(adaptations)
        plt.title("Najlepsze adaptacje w kolejnych iteracjach")
        plt.xlabel("Iteracje")
        plt.ylabel("Adaptacja")
    plot_path = os.path.join(plot_folder_path, f'all.png')
    plt.savefig(plot_path)
    plt.clf()


def simulate_PS(params):
    sim_stats = SimStat()

    iterations_limit = params['iterations_limit']
    stagnation_limit = params['stagnation_limit']

    max_iterations = params['max_iterations']
    acc = params['cost_function']['accuracy']
    min_y = params['cost_function']['min_y']

    for _ in range(params["repeats"]):
        best_adaptations:List[float] = []

        # Start PS
        swarm = Swarm(params)
        swarm.update_best_particle()

        best_particle = swarm.best_particle
        best_adaptations.append(best_particle.best_adaptation)

        stagnations = 0
        iterations = 0
        done = False
        while not done:
            swarm.update_swarm_pos()
            swarm.update_best_particle()
            # End PS

            if best_particle.best_adaptation == swarm.best_adaptation:
                stagnations += 1
                if stagnations == stagnation_limit:
                    sim_stats.stagnations_limit += 1
                    break
            else:
                stagnations = 0
                best_particle = swarm.best_particle

            best_adaptations.append(best_particle.best_adaptation)

            iterations += 1
            if isinstance(max_iterations, int):
                done = iterations >= max_iterations 
            else:
                done = abs(swarm.best_adaptation - min_y) < acc

            if iterations > iterations_limit:
                sim_stats.iteratioons_limit += 1
                break
        sim_stats.best_adaptations.append(best_adaptations)
        sim_stats.iterations.append(iterations)
    save_stats(params, sim_stats)


def simulate_DE(params):
    sim_stats = SimStat()

    iterations_limit = params['iterations_limit']
    stagnation_limit = params['stagnation_limit']

    max_iterations = params['max_iterations']
    acc = params['cost_function']['accuracy']
    min_y = params['cost_function']['min_y']

    x_range = [
        params['cost_function']['min_x'],
        params['cost_function']['max_x']
    ]
    equation = list(params['cost_function']['equation'].values())[0]

    for _ in range(params["repeats"]):
        best_adaptations:List[float] = []

        DE_algo = DE(
            params['population_size'],
            params['dimensions'],
            params['algorithm']['scaling_factor'],
            params['algorithm']['crossover_prob'],
            params['algorithm']['variant'],
            x_range,
            equation
        )

        input_x = DE_algo.vector_init()
        DE_algo.calc_cost(input_x)

        best_adaptations.append(DE_algo.best_cost)

        stagnations = 0
        iterations = 0
        done = False
        while not done:
            old_input_x = list(map(list, input_x))
            
            for i, x in enumerate(input_x):
                v = DE_algo.mutate(i, old_input_x)
                u = DE_algo.crossover(x, v)  # trial vector
                input_x[i] = DE_algo.selection(x, u)

            DE_algo.calc_cost(input_x)
            best_adaptations.append(DE_algo.best_cost)

            if best_adaptations[-1] == best_adaptations[-2]:
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
                done = abs(DE_algo.best_cost - min_y) < acc
            
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

    flag = False
    for i, params in enumerate(combinations_param_list):
        print(f'Simulation number: {i+1}')
        
        if params['algorithm']['method'] == 'DE':
            simulate_DE(params)
        elif params['algorithm']['method'] == 'PS':
            simulate_PS(params)


if __name__ == '__main__':
    main()
