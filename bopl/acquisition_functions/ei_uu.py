import torch

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bopl.utility.utility_model import UtilityModel


class ExpectedImprovementUnderUtilityUncertainty(AcquisitionFunction):
    r"""Expected Improvement under Utility Uncertainty.
    """

    def __init__(
        self,
        model: Model,
        utility_model: UtilityModel,
        Y: Tensor,
    ) -> None:
        r"""Expected Improvement under Utility Uncertainty.
        Args:
            model: A fitted BoTorch model.
            utility_model: A fitted utility model.
            Y: .
            menu_size: A positive integer number denoting the menu size.
        """
        super().__init__(model=model)
        self.utility_model = utility_model
        self.Y = Y
        self._attribute_sampler = SobolQMCNormalSampler(num_samples=32, collapse_batch_dims=True)
        self.register_buffer("max_utility_Y", self._compute_max_utility_Y())

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement under Utility Uncertainty on the candidate set `X`.
        Args:
            X: A `batch_shape x 1 x d`-dim Tensor.
        Returns:
            A `batch_shape'`-dim Tensor.
        """
        posterior = self.model.posterior(X)
        attribute_samples = self._attribute_sampler(posterior)
        utility_samples = self.utility_model(attribute_samples).squeeze(-1)
        max_utility_Y = self.max_utility_Y.expand(-1, utility_samples.shape[-2], utility_samples.shape[-1])
        improvement_samples = (utility_samples - max_utility_Y).clamp_min(0.0)
        expected_improvement = improvement_samples.mean(0).mean(0)
        return expected_improvement

    def _compute_max_utility_Y(self):
        utility_Y_samples = self.utility_model(self.Y)
        max_utility_Y_samples = utility_Y_samples.max(-1)[0]
        return max_utility_Y_samples.unsqueeze(dim=1).unsqueeze(dim=1)
