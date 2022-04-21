from typing import Dict


def get_equation(params:Dict):
    return list(params['cost_function']['equation'].values())[0]


def get_sigmoid(params:Dict):
    return {
        'exponent': params['algorithm']['exponent'],
        'max_val': params['algorithm']['max_val'],
        'min_val': params['algorithm']['min_val']
    } if params['algorithm']['use_sigmoid'] else {}


def get_params_BA(params:Dict) -> Dict:

    return {
        'population_size': params['population_size'],
        'f_range': (
            params['algorithm']['fmin'],
            params['algorithm']['fmax']
        ),
        'alpha': params['algorithm']['alpha'],
        'gamma': params['algorithm']['gamma'],
        'dimensions': params['dimensions'],
        'x_range' : (
            params['cost_function']['min_x'],
            params['cost_function']['max_x']
        ),
        'cost_function': get_equation(params),
        'sigmoid' : get_sigmoid(params)
    }


def get_params_BOA(params:Dict) -> Dict:

    return {
        'population_size': params['population_size'],
        'c': params['algorithm']['c'],
        'alpha': params['algorithm']['alpha'],
        'prob': params['algorithm']['prob'],
        'dimensions': params['dimensions'],
        'x_range' : (
            params['cost_function']['min_x'],
            params['cost_function']['max_x']
        ),
        'cost_function': get_equation(params),
        'sigmoid' : get_sigmoid(params)
    }
