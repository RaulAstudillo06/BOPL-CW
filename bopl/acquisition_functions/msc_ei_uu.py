import torch

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bopl.utility.utility_model import UtilityModel


class MenuSizeConstrainedExpectedImprovementUnderUtilityUncertainty(AcquisitionFunction):
    r"""Menu Size Constrained Expected Improvement under Utility Uncertainty.
    """

    def __init__(
        self,
        model: Model,
        utility_model: UtilityModel,
        Y: Tensor,
        menu_size: int = None,
    ) -> None:
        r"""Menu Size Constrained Expected Improvement under Utility Uncertainty.
        Args:
            model: A fitted BoTorch model.
            utility_model: A fitted utility model.
            Y: .
            menu_size: A positive integer number denoting the menu size.
        """
        super().__init__(model=model)
        self.utility_model = utility_model
        self.Y = Y
        self.menu_size = menu_size
        self._attribute_sampler = SobolQMCNormalSampler(num_samples=32, collapse_batch_dims=True)
        self.register_buffer("current_expected_utility", self._compute_current_expected_utility())
        self.register_buffer("aux", self._aux())

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
        utility_samples = self.utility_model(attribute_samples).expand(-1, -1, -1, self.aux.shape[-1])
        aux = self.aux.expand(-1, utility_samples.shape[1], utility_samples.shape[2], -1)
        utility_menus_samples = torch.maximum(utility_samples, aux)
        expected_utility_menus = utility_menus_samples.mean(0)
        expected_utility_best_menu = expected_utility_menus.max(-1)[0]
        improvement = (expected_utility_best_menu - self.current_expected_utility).clamp_min(0.0)
        expected_improvement = improvement.mean(0)
        return expected_improvement

    def _compute_current_expected_utility(self):
        utility_Y_samples = self.utility_model(self.Y)
        num_samples = utility_Y_samples.shape[0]
        utility_menus_samples = []
        for n in range(num_samples):
            utility_menus_samples.append(torch.combinations(utility_Y_samples[n, :], r=self.menu_size).max(-1)[0])
        
        expected_utility_menus = torch.stack(utility_menus_samples).mean(0)
        expected_utility_best_menu = expected_utility_menus.max(-1)[0]
        return expected_utility_best_menu

    def _aux(self):
        aux_menu_size = self.menu_size - 1 if self.menu_size > 1 else 1
        utility_Y_samples = self.utility_model(self.Y)
        num_samples = utility_Y_samples.shape[0]
        utility_menus_samples = []
        for n in range(num_samples):
            utility_menus_samples.append(torch.combinations(utility_Y_samples[n, :], r=aux_menu_size).max(-1)[0])
        
        return torch.stack(utility_menus_samples).unsqueeze(dim=1).unsqueeze(dim=1)
