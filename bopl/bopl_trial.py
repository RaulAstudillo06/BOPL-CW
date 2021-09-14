#!/usr/bin/env python3

from locale import Error
from typing import Callable, Dict, List, Optional

import numpy as np
import os
import sys
import time
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor

from bopl.acquisition_functions.ei_uu import ExpectedImprovementUnderUtilityUncertainty
from bopl.acquisition_functions.msc_ei_uu import MenuSizeConstrainedExpectedImprovementUnderUtilityUncertainty
from bopl.utils import (
    fit_model,
    generate_initial_design, 
    optimize_acqf_and_get_suggested_point
)
from bopl.utility.utility_model import UtilityModel

def bopl_trial(
    problem: str,
    get_objective_function: Callable,
    get_true_utility_function: Callable,
    utility_prior_model: UtilityModel,
    input_dim: int,
    menu_size: int,
    algo: str,
    algo_params: Dict,
    trial: int, 
    num_init_evals: int,
    num_bo_iter: float,
    restart: bool,
    ignore_failures: bool = False,
) -> None:
    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = script_dir[:-11]
    results_folder = project_path + "/experiments_results/" + problem + "/" + algo + "/"

    objective_function = get_objective_function(trial)
    true_utility_function = get_true_utility_function(trial)

    if restart:
        # Check if training data is already available
        try:
            # Current available evaluations
            X = torch.tensor(np.loadtxt(results_folder +
                                        "X/X_" + str(trial) + ".txt"))
            Y = torch.tensor(np.loadtxt(
                results_folder + "Y/Y_" + str(trial) + ".txt"))

            # Historical best observed objective values and running times
            util_of_historical_best_menus = list(np.loadtxt(
                results_folder + "util_of_historical_best_menus_" + str(trial) + ".txt"))
            runtimes = list(np.loadtxt(
                results_folder + "runtimes/runtimes_" + str(trial) + ".txt"))

            # Current best observed objective value
            util_of_current_best_menu = torch.tensor(util_of_historical_best_menus[-1])

            init_batch_id = len(util_of_historical_best_menus)
            print("Restarting experiment from available data.")

        except:

            # Initial evaluations
            X = generate_initial_design(
                num_samples=num_init_evals, input_dim=input_dim, seed=trial)
            Y = objective_function(X)

            # Utility of current best menu
            util_of_current_best_menu = compute_utility_of_best_menu(
                Y=Y, 
                true_utility_function=true_utility_function,
                utility_model=utility_prior_model,
                menu_size=menu_size,
            )

            # Utility of historical best menus
            util_of_historical_best_menus = [util_of_current_best_menu]
            runtimes = []

            init_batch_id = 1
    else:
        # Initial evaluations
        X = generate_initial_design(
            num_samples=num_init_evals, input_dim=input_dim, seed=trial)
        Y = objective_function(X)

        # Utility of current best menu
        util_of_current_best_menu = compute_utility_of_best_menu(
            Y=Y, 
            true_utility_function=true_utility_function,
            utility_model=utility_prior_model,
            menu_size=menu_size,
        )

        # Utility of historical best menus
        util_of_historical_best_menus = [util_of_current_best_menu]
        runtimes = []

        init_batch_id = 1


    for iteration in range(init_batch_id, num_bo_iter + 1):
        print("Problem: " + problem)
        print("Sampling policy: " + algo)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # New suggested point
        t0 = time.time()

        try:
            new_x = get_new_suggested_point(
                algo=algo,
                X=X,
                Y=Y,
                utility_prior_model=utility_prior_model,
                menu_size=menu_size,
                algo_params=algo_params,
            )
        except:
            if ignore_failures:
                print("An error ocurred when computing the next point to evaluate. Instead, a point will be chosen uniformly at random.")
                new_x = get_new_suggested_point(
                algo="Random",
                X=X,
                Y=Y,
                menu_size=menu_size,
                algo_params=algo_params,
            )
            else:
                raise Error("An error ocurred when computing the next point to evaluate.")


        t1 = time.time()
        runtimes.append(t1 - t0)

        # Evaluate objective at new point
        objective_new_x = objective_function(new_x)

        # Update training data
        X = torch.cat([X, new_x], 0)
        Y = torch.cat([Y, objective_new_x], 0)

        # Update utilities of historical best menus
        util_of_current_best_menu = compute_utility_of_best_menu(
            Y=Y, 
            true_utility_function=true_utility_function,
            utility_model=utility_prior_model,
            menu_size=menu_size,
        )

        util_of_historical_best_menus.append(util_of_current_best_menu)

        print("Utlity of current best menu: " + str(util_of_current_best_menu))

        # Save data
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        if not os.path.exists(results_folder + "runtimes/"):
            os.makedirs(results_folder + "runtimes/")
        if not os.path.exists(results_folder + "X/"):
            os.makedirs(results_folder + "X/")
        if not os.path.exists(results_folder + "Y/"):
            os.makedirs(results_folder + "Y/")

        np.savetxt(results_folder + "X/X_" + str(trial) + ".txt", X.numpy())
        np.savetxt(results_folder + "Y/Y_" +
                   str(trial) + ".txt", Y.numpy())
        np.savetxt(results_folder + "util_of_historical_best_menus_" +
                   str(trial) + ".txt", np.atleast_1d(util_of_historical_best_menus))
        np.savetxt(results_folder + "runtimes/runtimes_" +
                   str(trial) + ".txt", np.atleast_1d(runtimes))


def get_new_suggested_point(
    algo: str,
    X: Tensor,
    Y: Tensor,
    utility_prior_model: Callable,
    menu_size: int,
    algo_params: Optional[Dict] = None,
) -> Tensor:

    input_dim = X.shape[-1]

    if algo == "Random":
        return torch.rand([1, input_dim])
    elif algo == "MSC-EI-UU":
        model = fit_model(X=X, Y=Y, noiseless_obs=True)
        utility_model = utility_prior_model
        Y_nondominated = Y[is_non_dominated(Y)]
        effective_menu_size = min(menu_size, Y_nondominated.shape[0])
        acquisition_function = MenuSizeConstrainedExpectedImprovementUnderUtilityUncertainty(
            model=model, 
            utility_model=utility_model,
            Y=Y_nondominated,
            menu_size=effective_menu_size,
        )
    elif algo == "EI-UU":
        model = fit_model(X=X, Y=Y, noiseless_obs=True)
        utility_model = utility_prior_model
        acquisition_function = ExpectedImprovementUnderUtilityUncertainty(
            model=model, 
            utility_model=utility_model,
            Y=Y,
        )


    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])

    new_x = optimize_acqf_and_get_suggested_point(
        acq_func=acquisition_function,
        bounds=standard_bounds,
        batch_size=1,
    )

    return new_x


def compute_utility_of_best_menu(
    Y: Tensor,
    true_utility_function: Callable,
    utility_model: UtilityModel,
    menu_size: int,
) -> Tensor:
    Y_nondominated = Y[is_non_dominated(Y)]
    utility_Y_samples = utility_model(Y_nondominated)
    num_samples = utility_Y_samples.shape[0]
    effective_menu_size = min(menu_size, Y_nondominated.shape[0])
    utility_menus_samples = []

    for n in range(num_samples):
        utility_menus_samples.append(torch.combinations(utility_Y_samples[n, :], r=effective_menu_size).max(-1)[0])
    
    expected_utility_menus = torch.stack(utility_menus_samples).mean(0)
    expected_utility_best_menu = expected_utility_menus.max(-1)[0]
    #best_menu_id = expected_utility_menus.max(-1)[1]
    #best_menu_indices = torch.combinations(torch.tensor(range(Y_nondominated.shape[0])), r=effective_menu_size)[best_menu_id]
    #best_menu = Y_nondominated[best_menu_indices]
    #print(best_menu)
    #utility_best_menu = true_utility_function(best_menu).max()
    return expected_utility_best_menu