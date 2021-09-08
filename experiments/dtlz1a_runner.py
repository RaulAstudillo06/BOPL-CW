import os
import sys
import numpy as np
from numpy.random.mtrand import dirichlet
import torch

from botorch.settings import debug
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
output_dim = 2

def get_objective_function(seed: int) -> Callable:

    def g(X):
        gX = 100*(5 + torch.sum((X[..., 1:]-0.5)**2, axis=-1) - torch.sum(torch.cos(2*np.pi*(X[..., 1:] - 0.5)), axis=-1))
        return gX

    
    def f(X: Tensor) -> Tensor:
        fX = torch.empty(X.shape[:-1] + torch.Size([2]))
        fX[..., 0] = 0.5*X[..., 0]*(1 + g(X))
        fX[..., 1] = 0.5*(1 - X[..., 0])*(1 + g(X))
        fX = -fX
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
    num_samples=16,
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
algo = "Random"
algo_params = {}

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[1])

experiment_manager(
    problem = "dtlz1a",
    get_objective_function=get_objective_function,
    get_true_utility_function=get_true_utility_function,
    utility_prior_model=utility_prior_model,
    input_dim=input_dim,
    menu_size=3,
    algo=algo,
    algo_params=algo_params,
    first_trial=first_trial, 
    last_trial=last_trial,
    num_init_evals=2 * (input_dim + 1),
    num_bo_iter=50,
    restart=False,
    ignore_failures=False,
)