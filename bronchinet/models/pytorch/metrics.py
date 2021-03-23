
from typing import Tuple

from torch.nn import CrossEntropyLoss
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
                      'AirwayCompleteness',
                      'AirwayVolumeLeakage',
                      'AirwayCentrelineLeakage',
                      ]


class Metric(MetricBase, nn.Module):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(Metric, self).__init__(is_mask_exclude)
        nn.Module.__init__(self)

    def compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        if self._is_mask_exclude:
            return self._compute_masked(torch.flatten(y_true), torch.flatten(y_pred))
        else:
            return self._compute(torch.flatten(y_true), torch.flatten(y_pred))

    def forward(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return self.compute(y_true, y_pred)

    def loss(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return self.forward(y_true, y_pred)

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def _compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return self._compute(self._get_masked_input(y_true, y_true),
                             self._get_masked_input(y_pred, y_true))

    def _get_mask(self, y_true: torch.FloatTensor) -> torch.FloatTensor:
        return torch.where(y_true == self._value_mask_exclude, torch.zeros_like(y_true), torch.ones_like(y_true))

    def _get_masked_input(self, y_input: torch.FloatTensor, y_true: torch.FloatTensor) -> torch.FloatTensor:
        return torch.where(y_true == self._value_mask_exclude, torch.zeros_like(y_input), y_input)


class MetricWithUncertainty(Metric):
    # Composed uncertainty loss (ask Shuai)
    _epsilon_default = 0.01

    def __init__(self, metrics_loss: Metric, epsilon: float = _epsilon_default) -> None:
        self._metrics_loss = metrics_loss
        self._epsilon = epsilon
        super(MetricWithUncertainty, self).__init__(self._metrics_loss._is_mask_exclude)
        self._name_fun_out = self._metrics_loss._name_fun_out + '_uncertain'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return (1.0 - self._epsilon) * self._metrics_loss._compute(y_true, y_pred) + \
               self._epsilon * self._metrics_loss._compute(torch.ones_like(y_pred) / 3, y_pred)

    def _compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return (1.0 - self._epsilon) * self._metrics_loss.__compute_masked(y_true, y_pred) + \
               self._epsilon * self._metrics_loss.__compute_masked(torch.ones_like(y_pred) / 3, y_pred)


class CombineTwoMetrics(Metric):

    def __init__(self, metrics_1: Metric, metrics_2: Metric, weight_metric2over1: float = 1.0) -> None:
        super(CombineTwoMetrics, self).__init__(False)
        self._metrics_1 = metrics_1
        self._metrics_2 = metrics_2
        self._weight_metric2over1 = weight_metric2over1
        self._name_fun_out = '_'.join(['combi', metrics_1._name_fun_out, metrics_2._name_fun_out])

    def compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return self._metrics_1.compute(y_true, y_pred) + self._weight_metric2over1 * self._metrics_2.compute(y_true, y_pred)

    def forward(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return self._metrics_1.forward(y_true, y_pred) + self._weight_metric2over1 * self._metrics_2.forward(y_true, y_pred)


class MeanSquaredError(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MeanSquaredError, self).__init__(is_mask_exclude)
        self._name_fun_out  = 'mean_squared'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.mean(torch.square(y_pred - y_true))

    def _compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        mask = self._get_mask(y_true)
        return torch.mean(torch.square(y_pred - y_true) * mask)


class MeanSquaredErrorLogarithmic(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MeanSquaredErrorLogarithmic, self).__init__(is_mask_exclude)
        self._name_fun_out  = 'mean_squared_log'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.mean(torch.square(torch.log(torch.clip(y_pred, _EPS, None) + 1.0) -
                                       torch.log(torch.clip(y_true, _EPS, None) + 1.0)), axis=-1)

    def _compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        mask = self._get_mask(y_true)
        return torch.mean(torch.square(torch.log(torch.clip(y_pred, _EPS, None) + 1.0) -
                                       torch.log(torch.clip(y_true, _EPS, None) + 1.0)) * mask, axis=-1)


class BinaryCrossEntropy(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(BinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_fun_out = 'bin_cross'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.mean(- y_true * torch.log(y_pred + _EPS)
                          - (1.0 - y_true) * torch.log(1.0 - y_pred + _EPS))

    def _compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        mask = self._get_mask(y_true)
        return torch.mean((- y_true * torch.log(y_pred + _EPS)
                           - (1.0 - y_true) * torch.log(1.0 - y_pred + _EPS)) * mask)


class WeightedBinaryCrossEntropy(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(WeightedBinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_fun_out = 'weight_bin_cross'

    def _get_weights(self, y_true: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        num_class_1 = torch.count_nonzero(torch.where(y_true == 1.0, torch.ones_like(y_true), torch.zeros_like(y_true)), dtype=torch.int32)
        num_class_0 = torch.count_nonzero(torch.where(y_true == 0.0, torch.ones_like(y_true), torch.zeros_like(y_true)), dtype=torch.int32)
        return (1.0, torch.cast(num_class_0, dtype=torch.float32) / (torch.cast(num_class_1, dtype=torch.float32) + torch.variable(_EPS)))

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        weights = self._get_weights(y_true)
        return torch.mean(- weights[1] * y_true * torch.log(y_pred + _EPS)
                          - weights[0] * (1.0 - y_true) * torch.log(1.0 - y_pred + _EPS))

    def _compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        weights = self._get_weights(y_true)
        mask = self._get_mask(y_true)
        return torch.mean((- weights[1] * y_true * torch.log(y_pred + _EPS)
                           - weights[0] * (1.0 - y_true) * torch.log(1.0 - y_pred + _EPS)) * mask)


class WeightedBinaryCrossEntropyFixedWeights(WeightedBinaryCrossEntropy):
    weights_no_mask_exclude = (1.0, 80.0)
    weights_mask_exclude = (1.0, 300.0)  # for LUVAR data
    #weights_mask_exclude = (1.0, 361.0)  # for DLCST data

    def __init__(self, is_mask_exclude: bool = False) -> None:
        if is_mask_exclude:
            self._weights = self.weights_mask_exclude
        else:
            self._weights = self.weights_no_mask_exclude
        super(WeightedBinaryCrossEntropyFixedWeights, self).__init__(is_mask_exclude)
        self._name_fun_out = 'weight_bin_cross_fixed'

    def _get_weights(self, y_true: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return self._weights


class BinaryCrossEntropyFocalLoss(Metric):
    # Binary cross entropy + Focal loss
    _gamma_default = 2.0

    def __init__(self, gamma: float = _gamma_default, is_mask_exclude: bool = False) -> None:
        self._gamma = gamma
        super(BinaryCrossEntropyFocalLoss, self).__init__(is_mask_exclude)
        self._name_fun_out = 'bin_cross_focal_loss'

    def get_predprobs_classes(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        prob_1 = torch.where(y_true == 1.0, y_pred, torch.ones_like(y_pred))
        prob_0 = torch.where(y_true == 0.0, y_pred, torch.zeros_like(y_pred))
        return (prob_1, prob_0)

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.mean(- y_true * torch.pow(1.0 - y_pred, self._gamma) * torch.log(y_pred + _EPS)
                          - (1.0 - y_true) * torch.pow(y_pred, self._gamma) * torch.log(1.0 - y_pred + _EPS))

    def _compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        mask = self._get_mask(y_true)
        return torch.mean((- y_true * torch.pow(1.0 - y_pred, self._gamma) * torch.log(y_pred + _EPS)
                           - (1.0 - y_true) * torch.pow(y_pred, self._gamma) * torch.log(1.0 - y_pred + _EPS)) * mask)


class DiceCoefficient(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(DiceCoefficient, self).__init__(is_mask_exclude)
        self._name_fun_out = 'dice'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return (2.0 * torch.sum(y_true * y_pred)) / (torch.sum(y_true) + torch.sum(y_pred) + _SMOOTH)

    def forward(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return 1.0 - self.compute(y_true, y_pred)


class TruePositiveRate(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(TruePositiveRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'tp_rate'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sum(y_true * y_pred) / (torch.sum(y_true) + _SMOOTH)

    def forward(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return 1.0 - self.compute(y_true, y_pred)


class TrueNegativeRate(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(TrueNegativeRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'tn_rate'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sum((1.0 - y_true) * (1.0 - y_pred)) / (torch.sum((1.0 - y_true)) + _SMOOTH)

    def forward(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return 1.0 - self.compute(y_true, y_pred)


class FalsePositiveRate(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(FalsePositiveRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'fp_rate'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sum((1.0 - y_true) * y_pred) / (torch.sum((1.0 - y_true)) + _SMOOTH)


class FalseNegativeRate(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(FalseNegativeRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'fn_rate'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sum(y_true * (1.0 - y_pred)) / (torch.sum(y_true) + _SMOOTH)


class AirwayCompleteness(Metric):
    _is_use_ytrue_cenlines = True
    _is_use_ypred_cenlines = False

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayCompleteness, self).__init__(is_mask_exclude)
        self._name_fun_out = 'completeness'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sum(y_true * y_pred) / (torch.sum(y_true) + _SMOOTH)

    def forward(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return 1.0 - self.compute(y_true, y_pred)


class AirwayVolumeLeakage(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayVolumeLeakage, self).__init__(is_mask_exclude)
        self._name_fun_out = 'volume_leakage'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sum((1.0 - y_true) * y_pred) / (torch.sum(y_pred) + _SMOOTH)


class AirwayCentrelineLeakage(Metric):
    _is_use_ytrue_cenlines = False
    _is_use_ypred_cenlines = True

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayCentrelineLeakage, self).__init__(is_mask_exclude)
        self._name_fun_out = 'cenline_leakage'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sum((1.0 - y_true) * y_pred) / (torch.sum(y_pred) + _SMOOTH)