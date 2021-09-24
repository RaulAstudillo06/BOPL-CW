from abc import ABC
from typing import Any, Callable, Dict, List, Optional

import torch
from botorch.posteriors import Posterior
from torch import Tensor
from torch.nn import Module


class UtilityModel(Module, ABC):
    r"""Abstract base class for utility models.

    Args:
        _has_transformed_inputs: A boolean denoting whether `train_inputs` are currently
            stored as transformed or not.
        _original_train_inputs: A Tensor storing the original train inputs for use in
            `_revert_to_original_inputs`. Note that this is necessary since
            transform / untransform cycle introduces numerical errors which lead
            to upstream errors during training.
    """

    _has_transformed_inputs: bool = False
    _original_train_inputs: Optional[Tensor] = None


class GenericUtilityModel(UtilityModel):
    r"""A generic utility model constructed from a callable."""

    def __init__(
        self, 
        utility_function: Callable[[Tensor, Tensor], Tensor],
        prior_sampler: Callable[[int, int], Tensor], 
        num_samples: int,
        seed: Optional[int] = None,
    ) -> None:
        r"""A generic deterministic model constructed from a callable.
        Args:
            f: A callable mapping a `batch_shape x n x d`-dim input tensor `X`
                to a `batch_shape x n x m`-dimensional output tensor (the
                outcome dimension `m` must be explicit, even if `m=1`).
            num_outputs: The number of outputs `m`.
        """
        super().__init__()
        self._utility_function = utility_function
        self._prior_sampler = prior_sampler
        self.num_samples = num_samples
        self._seed = seed if seed is not None else torch.randint(0, 1000000, (1,)).item()
        self.register_buffer("posterior_samples", self._prior_sampler(self.num_samples, self._seed))

    def forward(self, Y: Tensor) -> Tensor:
        return self._utility_function(Y, self.posterior_samples)
