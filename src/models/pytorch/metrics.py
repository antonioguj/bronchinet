
from typing import Tuple

import torch.nn as nn
import torch

from models.metrics import MetricBase

_EPS = 1e-7
_SMOOTH = 1.0

LIST_AVAIL_METRICS = ['MeanSquaredError',
                      'MeanSquaredErrorLogarithmic',
                      'BinaryCrossEntropy',
                      'WeightedBinaryCrossEntropy',
                      'WeightedBinaryCrossEntropyFixedWeights',
                      'BinaryCrossEntropyFocalLoss',
                      'DiceCoefficient',
                      'TruePositiveRate',
                      'TrueNegativeRate',
                      'FalsePositiveRate',
                      'FalseNegativeRate',
                      ]


class Metric(MetricBase, nn.Module):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(Metric, self).__init__(is_mask_exclude)
        nn.Module.__init__(self)

    def compute(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        if self._is_mask_exclude:
            return self._compute_masked(torch.flatten(target), torch.flatten(input))
        else:
            return self._compute(torch.flatten(target), torch.flatten(input))

    def forward(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return self.compute(target, input)

    def loss(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return self.forward(target, input)

    def _compute(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _compute_masked(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return self._compute(self._get_masked_input(target, target),
                             self._get_masked_input(input, target))

    def _get_mask(self, target: torch.Tensor) -> torch.Tensor:
        return torch.where(target == self._value_mask_exclude, torch.zeros_like(target), torch.ones_like(target))

    def _get_masked_input(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.where(target == self._value_mask_exclude, torch.zeros_like(input), input)


class MetricWithUncertainty(Metric):
    # Composed uncertainty loss
    _epsilon_default = 0.01
    _num_classes_gt = 2

    def __init__(self, metrics_loss: Metric, epsilon: float = _epsilon_default) -> None:
        self._metrics_loss = metrics_loss
        self._epsilon = epsilon
        super(MetricWithUncertainty, self).__init__(self._metrics_loss._is_mask_exclude)
        self._name_fun_out = self._metrics_loss._name_fun_out + '_uncertain'

    def _compute(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return (1.0 - self._epsilon) * self._metrics_loss._compute(target, input) \
            + self._epsilon * self._metrics_loss._compute(torch.ones_like(input) / self._num_classes_gt, input)

    def _compute_masked(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return (1.0 - self._epsilon) * self._metrics_loss._compute_masked(target, input) \
            + self._epsilon * self._metrics_loss._compute_masked(torch.ones_like(input) / self._num_classes_gt, input)


class CombineTwoMetrics(Metric):

    def __init__(self, metrics_1: Metric, metrics_2: Metric, weight_metric2over1: float = 1.0) -> None:
        super(CombineTwoMetrics, self).__init__(False)
        self._metrics_1 = metrics_1
        self._metrics_2 = metrics_2
        self._weight_metric2over1 = weight_metric2over1
        self._name_fun_out = '_'.join(['combi', metrics_1._name_fun_out, metrics_2._name_fun_out])

    def compute(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return self._metrics_1.compute(target, input) \
            + self._weight_metric2over1 * self._metrics_2.compute(target, input)

    def forward(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return self._metrics_1.forward(target, input) \
            + self._weight_metric2over1 * self._metrics_2.forward(target, input)


class MeanSquaredError(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MeanSquaredError, self).__init__(is_mask_exclude)
        self._name_fun_out = 'mean_squared'

    def _compute(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.square(input - target))

    def _compute_masked(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        mask = self._get_mask(target)
        return torch.mean(torch.square(input - target) * mask)


class MeanSquaredErrorLogarithmic(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MeanSquaredErrorLogarithmic, self).__init__(is_mask_exclude)
        self._name_fun_out = 'mean_squared_log'

    def _compute(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.square(torch.log(torch.clip(input, _EPS, None) + 1.0)
                                       - torch.log(torch.clip(target, _EPS, None) + 1.0)))

    def _compute_masked(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        mask = self._get_mask(target)
        return torch.mean(torch.square(torch.log(torch.clip(input, _EPS, None) + 1.0)
                                       - torch.log(torch.clip(target, _EPS, None) + 1.0)) * mask)


class BinaryCrossEntropy(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(BinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_fun_out = 'bin_cross'

    def _compute(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return torch.mean(- target * torch.log(input + _EPS)
                          - (1.0 - target) * torch.log(1.0 - input + _EPS))

    def _compute_masked(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        mask = self._get_mask(target)
        return torch.mean((- target * torch.log(input + _EPS)
                           - (1.0 - target) * torch.log(1.0 - input + _EPS)) * mask)


class WeightedBinaryCrossEntropy(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(WeightedBinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_fun_out = 'weight_bin_cross'

    def _get_weights(self, target: torch.Tensor) -> Tuple[float, float]:
        num_class_1 = torch.count_nonzero(torch.where(target == 1.0, torch.ones_like(target), torch.zeros_like(target)),
                                          dtype=torch.int32)
        num_class_0 = torch.count_nonzero(torch.where(target == 0.0, torch.ones_like(target), torch.zeros_like(target)),
                                          dtype=torch.int32)
        return (1.0, torch.cast(num_class_0, dtype=torch.float32)
                / (torch.cast(num_class_1, dtype=torch.float32) + torch.variable(_EPS)))

    def _compute(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        weights = self._get_weights(target)
        return torch.mean(- weights[1] * target * torch.log(input + _EPS)
                          - weights[0] * (1.0 - target) * torch.log(1.0 - input + _EPS))

    def _compute_masked(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        weights = self._get_weights(target)
        mask = self._get_mask(target)
        return torch.mean((- weights[1] * target * torch.log(input + _EPS)
                           - weights[0] * (1.0 - target) * torch.log(1.0 - input + _EPS)) * mask)


class WeightedBinaryCrossEntropyFixedWeights(WeightedBinaryCrossEntropy):
    weights_no_mask_exclude = (1.0, 80.0)
    weights_mask_exclude = (1.0, 300.0)  # for LUVAR data
    # weights_mask_exclude = (1.0, 361.0)  # for DLCST data

    def __init__(self, is_mask_exclude: bool = False) -> None:
        if is_mask_exclude:
            self._weights = self.weights_mask_exclude
        else:
            self._weights = self.weights_no_mask_exclude
        super(WeightedBinaryCrossEntropyFixedWeights, self).__init__(is_mask_exclude)
        self._name_fun_out = 'weight_bin_cross_fixed'

    def _get_weights(self, target: torch.Tensor) -> Tuple[float, float]:
        return self._weights


class BinaryCrossEntropyFocalLoss(Metric):
    # Binary cross entropy + Focal loss
    _gamma_default = 2.0

    def __init__(self, gamma: float = _gamma_default, is_mask_exclude: bool = False) -> None:
        self._gamma = gamma
        super(BinaryCrossEntropyFocalLoss, self).__init__(is_mask_exclude)
        self._name_fun_out = 'bin_cross_focal_loss'

    def get_predprobs_classes(self, target: torch.Tensor, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prob_1 = torch.where(target == 1.0, input, torch.ones_like(input))
        prob_0 = torch.where(target == 0.0, input, torch.zeros_like(input))
        return (prob_1, prob_0)

    def _compute(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return torch.mean(- target * torch.pow(1.0 - input, self._gamma) * torch.log(input + _EPS)
                          - (1.0 - target) * torch.pow(input, self._gamma) * torch.log(1.0 - input + _EPS))

    def _compute_masked(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        mask = self._get_mask(target)
        return torch.mean((- target * torch.pow(1.0 - input, self._gamma) * torch.log(input + _EPS)
                           - (1.0 - target) * torch.pow(input, self._gamma) * torch.log(1.0 - input + _EPS)) * mask)


class DiceCoefficient(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(DiceCoefficient, self).__init__(is_mask_exclude)
        self._name_fun_out = 'dice'

    def _compute(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return (2.0 * torch.sum(target * input)) / (torch.sum(target) + torch.sum(input) + _SMOOTH)

    def forward(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.compute(target, input)


class TruePositiveRate(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(TruePositiveRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'tp_rate'

    def _compute(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return torch.sum(target * input) / (torch.sum(target) + _SMOOTH)

    def forward(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.compute(target, input)


class TrueNegativeRate(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(TrueNegativeRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'tn_rate'

    def _compute(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return torch.sum((1.0 - target) * (1.0 - input)) / (torch.sum((1.0 - target)) + _SMOOTH)

    def forward(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.compute(target, input)


class FalsePositiveRate(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(FalsePositiveRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'fp_rate'

    def _compute(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return torch.sum((1.0 - target) * input) / (torch.sum((1.0 - target)) + _SMOOTH)


class FalseNegativeRate(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(FalseNegativeRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'fn_rate'

    def _compute(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return torch.sum(target * (1.0 - input)) / (torch.sum(target) + _SMOOTH)
