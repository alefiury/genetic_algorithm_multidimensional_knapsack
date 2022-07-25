import operator
import functools
from typing import Tuple, List
from dataclasses import dataclass
import copy

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass(frozen=True)
class Colors(metaclass=Singleton):
    GREEN: str = '\033[32m'
    BLUE: str = '\033[34m'
    CYAN: str = '\033[36m'
    WHITE: str = '\033[37m'
    RESET: str = '\033[0m'


@dataclass(frozen=True)
class LogFormatter(metaclass=Singleton):
    colors_single = Colors()
    TIME_DATA: str = colors_single.BLUE + '%(asctime)s' + colors_single.RESET
    MODULE_NAME: str = colors_single.CYAN + '%(module)s' + colors_single.RESET
    LEVEL_NAME: str = colors_single.GREEN + '%(levelname)s' + colors_single.RESET
    MESSAGE: str = colors_single.WHITE + '%(message)s' + colors_single.RESET
    FORMATTER = '['+TIME_DATA+']'+'['+MODULE_NAME+']'+'['+LEVEL_NAME+']'+' - '+MESSAGE


formatter_single = LogFormatter()

def split_metadata_first_problem(metadata_path: str) -> None:
    with open(metadata_path, 'r') as file:
        lines = [line.rstrip().lstrip().split(' ') for line in file.readlines()]

    values = functools.reduce(operator.iconcat, lines, [])
    values = [int(value) for value in values]

    num_problems = values [0]

    del values[0]

    num_vars = values[0]
    num_const = values[1]
    opt = values[2]

    consts = values[3: num_vars+3]
    weights = values[num_vars+3:num_vars*num_const + num_vars+3]
    c = values[num_vars*num_const + num_vars+3:num_vars*num_const + num_vars+3 + num_const]

    return num_vars, num_const, opt, np.array(consts), np.array(weights).reshape((num_const, num_vars)), np.array(c)

def split_metadata(metadata_path: str) -> None:
    problems_l = []
    problem_dict = dict()
    with open(metadata_path, 'r') as file:
        lines = [line.rstrip().lstrip().split(' ') for line in file.readlines()]

    values = functools.reduce(operator.iconcat, lines, [])
    values = [int(value) for value in values]

    num_problems = values [0]

    del values[0]

    num_vars = values[0]
    num_const = values[1]
    opt = values[2]

    for problem in range(num_problems):
        nums = values[(problem*(num_vars+(num_const*num_vars)+num_const+3)):((problem+1)*(num_vars+(num_const*num_vars)+num_const+3))]
        num_variables = nums[0]
        num_constraints = nums[1]
        opt = nums[2]
        coeficients = nums[3: num_vars+3]
        constraints = nums[num_vars+3:num_vars*num_const + num_vars+3]
        max_weigths = nums[num_vars*num_const + num_vars+3:num_vars*num_const + num_vars+3 + num_const]

        problem_dict = {
            "num_variables": num_variables,
            "num_constraints": num_constraints,
            "coeficients": np.array(coeficients),
            "constraints": np.array(constraints).reshape((num_constraints, num_vars)),
            "max_weigths": np.array(max_weigths)
        }
        problems_l.append(problem_dict.copy())
        problem_dict.clear()


    return problems_l


def write_results(problem_name: str, best_fitness: int, mean_best_fitness: int, best_individual: List, output_file_path: str) -> None:
    with open(output_file_path, 'a') as out_file:
        out_file.write(f"{problem_name}|{best_fitness}|{mean_best_fitness}|{best_individual}\n")