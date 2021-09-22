import os
import sys
import torch

from botorch.settings import debug
from math import pi
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
input_dim = 5
output_dim = 4
k = 2


def get_objective_function(seed: int) -> Callable:
    
    def f(X: Tensor) -> Tensor:
        gX = torch.zeros(size=X.shape[:-1])
        fX = torch.ones(size=X.shape[:-1] + torch.Size([output_dim]))
        for i in range(k):
            gX += torch.square(X[..., input_dim - i - 1] - 0.5)
        for i in range(output_dim - 1):
            fX[..., 0] *= torch.cos(0.5 * pi * X[..., i])
        for i in range(output_dim - 2):
            fX[..., 1] *= torch.cos(0.5 * pi * X[..., i])
        fX[..., 1] *= torch.sin(0.5 * pi * X[..., output_dim - 2])
        fX[..., 2] = torch.cos(0.5 * pi * X[..., 0]) * torch.sin(0.5 * pi * X[..., 1])
        fX[..., 3] = torch.sin(0.5 * pi * X[..., 0])
        for j in range(output_dim):
            fX[..., j] *= (1 + gX)
        fX = -fX
        return fX
        
    return f

# Utility's prior distribution
def utility_function(Y: Tensor, parameter: Tensor) -> Tensor:
    if Y.ndim == 2:
        aux = torch.einsum('ac, bc -> abc', parameter, Y)
    elif Y.ndim == 4:
        aux = torch.einsum('ae, bcde -> abcde', parameter, Y)
    utility = aux.min(dim=-1)[0] + 0.05 * aux.sum(dim=-1)
    return utility

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
algo = "MSC-EI-UU"
algo_params = {}

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[1])

experiment_manager(
    problem = "dtlz2",
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
