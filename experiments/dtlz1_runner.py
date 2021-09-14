import os
import sys
import numpy as np
from numpy.random.mtrand import dirichlet
import torch

from botorch.settings import debug
from botorch.test_functions.multi_objective import DTLZ1
from torch import Tensor
from typing import Callable

torch.set_default_dtype(torch.float64)
debug._set_state(True)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from bopl.experiment_manager import experiment_manager
from bopl.utility.utility_model import GenericUtilityModel


# Objective function
input_dim = 6
output_dim = 4

dtlz1 = DTLZ1(dim=input_dim, num_objectives=output_dim, negate=True)

def get_objective_function(seed: int) -> Callable:
    
    def f(X: Tensor) -> Tensor:
        fX = dtlz1(X)
        return fX
        
    return f

# Utility's prior distribution
def utility_function(Y: Tensor, parameter: Tensor) -> Tensor:
    return torch.inner(parameter, Y)

def utility_prior_sampler(num_samples: int = 1, seed: int =None) -> Tensor:
    dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(output_dim))
    if seed is None:
        samples = dirichlet.sample_n(num_samples)
    else:
        original_random_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        samples = dirichlet.sample_n(num_samples)
        torch.random.set_rng_state(original_random_state)

    return samples

utility_prior_model = GenericUtilityModel(
    utility_function=utility_function, 
    prior_sampler=utility_prior_sampler,
    num_samples=32,
    seed=1,
)

# True underlying utility
def get_true_utility_function(seed:int) -> Callable:
    true_utility_parameter = utility_prior_sampler(num_samples=1, seed=seed)
    print(true_utility_parameter)
    def true_utility_function(Y: Tensor) -> Tensor:
        return utility_function(Y, true_utility_parameter)

    return true_utility_function


# Algos
algo = "EI-UU"
algo_params = {}

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[1])

experiment_manager(
    problem = "dtlz1",
    get_objective_function=get_objective_function,
    get_true_utility_function=get_true_utility_function,
    utility_prior_model=utility_prior_model,
    input_dim=input_dim,
    menu_size=6,
    algo=algo,
    algo_params=algo_params,
    first_trial=first_trial, 
    last_trial=last_trial,
    num_init_evals=2 * (input_dim + 1),
    num_bo_iter=60,
    restart=True,
    ignore_failures=False,
)
