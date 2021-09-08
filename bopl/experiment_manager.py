from typing import Callable, Dict, List, Optional

import os
import sys

from bopl.bopl_trial import bopl_trial
from bopl.utility.utility_model import UtilityModel


def experiment_manager(
    problem: str,
    get_objective_function: Callable,
    get_true_utility_function: Callable,
    utility_prior_model: UtilityModel,
    input_dim: int,
    menu_size: int,
    algo: str,
    algo_params: Dict,
    first_trial: int, 
    last_trial: int,
    num_init_evals: int,
    num_bo_iter: float,
    restart: bool,
    ignore_failures: bool = False,
) -> None:

    for trial in range(first_trial, last_trial + 1):

        objective_function = get_objective_function(seed=trial)

        bopl_trial(
            problem=problem,
            get_objective_function=get_objective_function,
            get_true_utility_function=get_true_utility_function,
            utility_prior_model=utility_prior_model,
            input_dim=input_dim,
            menu_size=menu_size,
            algo=algo,
            algo_params=algo_params,
            trial=trial, 
            num_init_evals=num_init_evals,
            num_bo_iter=num_bo_iter,
            restart=restart,
            ignore_failures=ignore_failures,
        )
