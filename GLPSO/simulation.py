from typing import Union
from glob import glob
import matplotlib.pyplot as plt
import params as parameters
from simstat import SimStat
from typing import List, Dict
from itertools import product
from GLPSO import GLPSO
from GLPSO_OBL import GLPSO_OBL
from GLPSO_DE import GLPSO_DE
from multiprocessing import Process
import multiprocessing as mp
import statistics
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
    print(exp_name)
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

        "min_x": params['cost_function']['min_x'],
        "max_x": params['cost_function']['max_x'],

        "stagnation_limit": sim_stats.stagnations_limit,

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

    for _, adaptations in enumerate(sim_stats.best_adaptations):
        plt.plot(adaptations)
        plt.title("Najlepsze adaptacje w kolejnych iteracjach")
        plt.xlabel("Iteracje")
        plt.ylabel("Adaptacja")
    plot_path = os.path.join(plot_folder_path, f'all.png')
    plt.savefig(plot_path)
    plt.clf()


def simulate_GLPSO(params, GLPSO_algo:Union[GLPSO, GLPSO_OBL, GLPSO_DE]):
    sim_stats = SimStat()
    stagnation_limit = params['stagnation_limit']
    max_iterations = params['max_iterations']

    for repeat in range(params['repeats']):
        if (repeat+1) % 5 == 0:
            print(f'{repeat+1}/{params["repeats"]}')
        best_adaptations:List[float] = []
        best_adaptations.append(GLPSO_algo.best_eval)

        stagnations = 0
        iterations = 0
        done = False
        while not done:
            GLPSO_algo.do_epoch()
            best_adaptations.append(GLPSO_algo.best_eval)

            if best_adaptations[-1] == best_adaptations[-2]:
                stagnations += 1
                if stagnations == stagnation_limit:
                    sim_stats.stagnations_limit += 1
                    break
            else:
                stagnations = 0

            iterations += 1
            done = iterations >= max_iterations 
        sim_stats.best_adaptations.append(best_adaptations)
        sim_stats.iterations.append(iterations)
        GLPSO_algo = GLPSO_algo.__class__(params)
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
        processes:List[Process] = []
        
        print("Start GLPSO")
        params['algorithm']['method'] = 'GLPSO'
        p = Process(target=simulate_GLPSO, args=(params, GLPSO(params)))
        p.start()
        processes.append(p)
        print("End GLPSO")

        print("Start GLPSO_OBL")
        params['algorithm']['method'] = 'GLPSO_OBL'
        p = Process(target=simulate_GLPSO, args=(params, GLPSO_OBL(params)))
        p.start()
        processes.append(p)
        print("End GLPSO_OBL")

        print("Start GLPSO_DE")
        params['algorithm']['method'] = 'GLPSO_DE'
        p = Process(target=simulate_GLPSO, args=(params, GLPSO_DE(params)))
        p.start()
        processes.append(p)
        print("End GLPSO_DE")
        
        for p in processes:
            p.join()


if __name__ == '__main__':
    main()
