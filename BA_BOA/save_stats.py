import matplotlib.pyplot as plt
from typing import List, Dict
from simstat import SimStat

import statistics
import glob
import json
import os


OUTPUT_FOLDER = "output"


def save_stats(params, sim_stats:SimStat):
    equation_name = list(params['cost_function']['equation'].keys())[0]
    exp_name = f"{equation_name}_{params['algorithm']['method']}_E{params['max_iterations']}"
    exp_name += f"_P{params['population_size']}_D{params['dimensions']}"

    algo_dict:Dict = params['algorithm']
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
        folders_count = len(glob.glob(directory_path+'*'))
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
