from typing import Callable, Dict, Optional

import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ScalarizedObjective
from botorch.generation.gen import get_best_candidates
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.outcome import (
    Standardize,
)
from botorch.optim.optimize import optimize_acqf
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.utils.transforms import normalize
from gpytorch.constraints.constraints import Interval
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor


def fit_model(
    X: Tensor,
    Y: Tensor,
    noiseless_obs: bool = False,
):
    
    # Add feature dimension if necessary
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)

    # Outcome transform
    outcome_transform = Standardize(m=Y.shape[-1], batch_shape=Y.shape[:-2])
    
    # Define model
    if noiseless_obs:
        model = FixedNoiseGP(
            train_X=X,
            train_Y=Y,
            train_Yvar=torch.ones(size=Y.shape) * 1e-6,
            outcome_transform=outcome_transform,
        )    
    else:
        model = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            outcome_transform=outcome_transform,
        )

    # Train model
    model.outcome_transform.eval()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    return model


def generate_initial_design(
    num_samples: int,
    input_dim: int,
    sobol: bool=True, 
    seed: int=None
    ) -> Tensor:
    """Generates initial training inputs."""
    if sobol:
        soboleng = torch.quasirandom.SobolEngine(dimension=input_dim, scramble=True, seed=seed)
        X = soboleng.draw(num_samples).to(dtype=torch.double)
    else:
        if seed is not None:
            old_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
            X = torch.rand([num_samples, input_dim])
            torch.random.set_rng_state(old_state)
        else:
            X = torch.rand([num_samples, input_dim])
    return X


def optimize_acqf_and_get_suggested_point(
    acq_func,
    bounds,
    batch_size,
    posterior_mean=None,
    ) -> Tensor:
    """Optimizes the acquisition function, and returns a new candidate."""
    input_dim = bounds.shape[1]
    num_restarts=10*input_dim
    raw_samples=100*input_dim
    options={"batch_limit": 5}
    
    if posterior_mean is not None:
        baseline_candidate, _ = optimize_acqf(
            acq_function=posterior_mean,
            bounds=bounds,
            q=batch_size,
            num_restarts=10*input_dim,
            raw_samples=100*input_dim,
            options=options,
        )
    
        baseline_candidate = baseline_candidate.detach().view(torch.Size([1, batch_size, input_dim]))
    else:
        baseline_candidate = None

    if baseline_candidate is None:
        batch_initial_conditions = gen_batch_initial_conditions(
                acq_function=acq_func,
                bounds=bounds,
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
            )
    else:
        batch_initial_conditions = gen_batch_initial_conditions(
                acq_function=acq_func,
                bounds=bounds,
                q=batch_size,
                num_restarts=num_restarts - 1,
                raw_samples=raw_samples,
                options=options,
            )
        
        batch_initial_conditions = torch.cat([batch_initial_conditions, baseline_candidate], 0)

    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        options=options,
    )
    if baseline_candidate is not None:
        baseline_acq_value = acq_func.forward(baseline_candidate)[0].detach()

        if baseline_acq_value >= acq_value:
            print('Baseline candidate was best found.')
            print(acq_value)
            print(baseline_acq_value)
            candidate = baseline_candidate
            
    new_x = candidate.detach().view([batch_size, input_dim])
    return new_x
