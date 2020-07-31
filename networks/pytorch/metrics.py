
from typing import Tuple
import numpy as np

from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch

from networks.metrics import Metrics as MetricsBase

_eps = 1e-7
_smooth = 1.0


class Metrics(MetricsBase, nn.Module):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(Metrics, self).__init__(is_mask_exclude)

    def compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        if self.is_mask_exclude:
            return self.__compute_masked(torch.flatten(y_true), torch.flatten(y_pred.flatten))
        else:
            return self._compute(torch.flatten(y_true), torch.flatten(y_pred))

    def forward(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return self.compute(y_true, y_pred)

    def loss(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return self.forward(y_true, y_pred)

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def __compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return self._compute(self._get_masked_input(y_true, y_true),
                             self._get_masked_input(y_pred, y_true))

    def get_mask(self, y_true: torch.FloatTensor) -> torch.FloatTensor:
        return torch.where(y_true == self._value_mask_exclude, torch.zeros_like(y_true), torch.ones_like(y_true))

    def get_masked_input(self, y_input: torch.FloatTensor, y_true: torch.FloatTensor) -> torch.FloatTensor:
        return torch.where(y_true == self._value_mask_exclude, torch.zeros_like(y_input), y_input)


class MetricsWithUncertaintyLoss(Metrics):
    # composed uncertainty loss (ask Shuai)
    _epsilon_default = 0.01

    def __init__(self, metrics_loss: Metrics, epsilon: float = _epsilon_default) -> None:
        self._metrics_loss = metrics_loss
        self._epsilon = epsilon
        super(MetricsWithUncertaintyLoss, self).__init__(self._metrics_loss._is_mask_exclude)
        self._name_func_out = self._metrics_loss._name_func_out + '_uncerloss'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return (1.0 - self._epsilon) * self._metrics_loss._compute(y_true, y_pred) + \
               self._epsilon * self._metrics_loss._compute(torch.ones_like(y_pred) / 3, y_pred)

    def __compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return (1.0 - self._epsilon) * self._metrics_loss.__compute_masked(y_true, y_pred) + \
               self._epsilon * self._metrics_loss.__compute_masked(torch.ones_like(y_pred) / 3, y_pred)


class MeanSquaredError(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MeanSquaredError, self).__init__(is_mask_exclude)
        self._name_func_out  = 'mean_squared'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.mean(torch.square(y_pred - y_true))

    def _compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        mask = self._get_mask(y_true)
        return torch.mean(torch.square(y_pred - y_true) * mask)


class MeanSquaredErrorLogarithmic(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MeanSquaredErrorLogarithmic, self).__init__(is_mask_exclude)
        self._name_func_out  = 'mean_squared_log'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.mean(torch.square(torch.log(torch.clip(y_pred, _eps, None) + 1.0) -
                                       torch.log(torch.clip(y_true, _eps, None) + 1.0)), axis=-1)

    def _compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        mask = self._get_mask(y_true)
        return torch.mean(torch.square(torch.log(torch.clip(y_pred, _eps, None) + 1.0) -
                                       torch.log(torch.clip(y_true, _eps, None) + 1.0)) * mask, axis=-1)


class BinaryCrossEntropy(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(BinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_func_out = 'bin_cross'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.mean(- y_true * torch.log(y_pred + _eps)
                          - (1.0 - y_true) * torch.log(1.0 - y_pred + _eps))

    def _compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        mask = self._get_mask(y_true)
        return torch.mean((- y_true * torch.log(y_pred +_eps)
                           - (1.0 - y_true) * torch.log(1.0 - y_pred +_eps)) * mask)


class WeightedBinaryCrossEntropy(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(WeightedBinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_func_out = 'weight_bin_cross'

    def _get_weights(self, y_true: torch.FloatTensor) -> Tuple[torch.Float, torch.Float]:
        num_class_1 = torch.count_nonzero(torch.where(y_true == 1.0, torch.ones_like(y_true), torch.zeros_like(y_true)), dtype=torch.int32)
        num_class_0 = torch.count_nonzero(torch.where(y_true == 0.0, torch.ones_like(y_true), torch.zeros_like(y_true)), dtype=torch.int32)
        return (1.0, torch.cast(num_class_0, dtype=torch.float32) / (torch.cast(num_class_1, dtype=torch.float32) + torch.variable(_eps)))

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        weights = self._get_weights(y_true)
        return torch.mean(- weights[1] * y_true * torch.log(y_pred + _eps)
                          - weights[0] * (1.0 - y_true) * torch.log(1.0 - y_pred + _eps))

    def _compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        weights = self._get_weights(y_true)
        mask = self._get_mask(y_true)
        return torch.mean((- weights[1] * y_true * torch.log(y_pred + _eps)
                           - weights[0] * (1.0 - y_true) * torch.log(1.0 - y_pred + _eps)) * mask)


class WeightedBinaryCrossEntropyFixedWeights(Metrics):
    weights_no_mask_exclude = [1.0, 80.0]
    weights_mask_exclude = [1.0, 300.0]  # for LUVAR data
    #weights_mask_exclude = [1.0, 361.0]  # for DLCST data

    def __init__(self, is_mask_exclude: bool = False) -> None:
        if is_mask_exclude:
            self._weights = self.weights_mask_exclude
        else:
            self._weights = self.weights_no_mask_exclude
        super(WeightedBinaryCrossEntropyFixedWeights, self).__init__(is_mask_exclude)
        self._name_func_out = 'weight_bin_cross_fixed'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.mean(- self._weights[1] * y_true * torch.log(y_pred + _eps)
                          - self._weights[0] * (1.0 - y_true) * torch.log(1.0 - y_pred + _eps))

    def _compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        mask = self._get_mask(y_true)
        return torch.mean((- self._weights[1] * y_true * torch.log(y_pred + _eps)
                           - self._weights[0] * (1.0 - y_true) * torch.log(1.0 - y_pred + _eps)) * mask)


class BinaryCrossEntropyFocalLoss(Metrics):
    # Binary cross entropy + Focal loss
    _gamma_default = 2.0

    def __init__(self, gamma: float = _gamma_default, is_mask_exclude: bool = False) -> None:
        self._gamma = gamma
        super(BinaryCrossEntropyFocalLoss, self).__init__(is_mask_exclude)
        self._name_func_out = 'bin_cross_focal_loss'

    def get_predprobs_classes(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        prob_1 = torch.where(y_true == 1.0, y_pred, torch.ones_like(y_pred))
        prob_0 = torch.where(y_true == 0.0, y_pred, torch.zeros_like(y_pred))
        return (prob_1, prob_0)

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return torch.mean(- y_true * torch.pow(1.0 - y_pred, self._gamma) * torch.log(y_pred + _eps)
                          - (1.0 - y_true) * torch.pow(y_pred, self._gamma) * torch.log(1.0 - y_pred + _eps))

    def _compute_masked(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        mask = self._get_mask(y_true)
        return torch.mean((- y_true * torch.pow(1.0 - y_pred, self._gamma) * torch.log(y_pred + _eps)
                           - (1.0 - y_true) * torch.pow(y_pred, self._gamma) * torch.log(1.0 - y_pred + _eps)) * mask)


# Dice coefficient
class DiceCoefficient(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(DiceCoefficient, self).__init__(is_mask_exclude)
        self._name_func_out = 'dice'

    def _compute(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return (2.0 * torch.sum(y_true * y_pred)) / (torch.sum(y_true) + torch.sum(y_pred) + _smooth)

    def forward(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        return 1.0 - self.compute(y_true, y_pred)


# combination of two metrics
class CombineLossTwoMetrics(Metrics):
    weights_metrics = [1.0, 3.0]

    def __init__(self, metrics1, metrics2, is_mask_exclude= False):
        super(CombineLossTwoMetrics, self).__init__(is_mask_exclude)
        self.metrics1 = metrics1
        self.metrics2 = metrics2
        self._name_func_out = '_'.join(['comb', metrics1._name_func_out, metrics2._name_func_out])

    def loss(self, y_true, y_pred):
        return self.weights_metrics[0] * self.metrics1.loss(y_true, y_pred) + \
               self.weights_metrics[1] * self.metrics2.loss(y_true, y_pred)



# all available metrics
def DICTAVAILMETRICLASS(option,
                        is_mask_exclude= False):
    list_metric_avail = ['MeanSquared',
                         'BinaryCrossEntropy', 'WeightedBinaryCrossEntropy',
                         'DiceCoefficient',
                         'TruePositiveRate', 'TrueNegativeRate', 'FalsePositiveRate', 'FalseNegativeRate',
                         'AirwayCompleteness', 'AirwayVolumeLeakage',
                         'AirwayCompletenessModified', 'AirwayCentrelineLeakage',
                         'AirwayCentrelineFalsePositiveDistanceError', 'AirwayCentrelineFalseNegativeDistanceError']

    if   (option == 'MeanSquared'):
        return MeanSquaredError(is_mask_exclude= is_mask_exclude)
    elif (option == 'BinaryCrossEntropy'):
        return BinaryCrossEntropy(is_mask_exclude= is_mask_exclude)
    elif (option == 'WeightedBinaryCrossEntropy'):
        return WeightedBinaryCrossEntropyFixedWeights(is_mask_exclude=is_mask_exclude)
    elif (option == 'WeightedBinaryCrossEntropyFixedWeights'):
        return WeightedBinaryCrossEntropyFixedWeights(is_mask_exclude=is_mask_exclude)
    elif (option == 'DiceCoefficient'):
        return DiceCoefficient(is_mask_exclude= is_mask_exclude)
    elif (option == 'TruePositiveRate'):
        return TruePositiveRate(is_mask_exclude=is_mask_exclude)
    elif (option == 'TrueNegativeRate'):
        return TrueNegativeRate(is_mask_exclude=is_mask_exclude)
    elif (option == 'FalsePositiveRate'):
        return FalsePositiveRate(is_mask_exclude=is_mask_exclude)
    elif (option == 'FalseNegativeRate'):
        return FalseNegativeRate(is_mask_exclude=is_mask_exclude)
    elif (option == 'AirwayCompleteness'):
        return AirwayCompleteness(is_mask_exclude=is_mask_exclude)
    elif (option == 'AirwayVolumeLeakage'):
        return AirwayVolumeLeakage(is_mask_exclude=is_mask_exclude)
    elif (option == 'AirwayCompletenessModified'):
        return AirwayCompletenessModified(is_mask_exclude=is_mask_exclude)
    elif (option == 'AirwayCentrelineLeakage'):
        return AirwayCentrelineLeakage(is_mask_exclude=is_mask_exclude)
    elif (option == 'AirwayCentrelineFalsePositiveDistanceError'):
        return AirwayCentrelineFalsePositiveDistanceError(is_mask_exclude=is_mask_exclude)
    elif (option == 'AirwayCentrelineFalseNegativeDistanceError'):
        return AirwayCentrelineFalseNegativeDistanceError(is_mask_exclude=is_mask_exclude)
    else:
        message = 'Metric \'%s\' chosen not found. Metrics available: \'%s\'...' %(option, ', '.join(list_metric_avail))
        catch_error_exception(message)
        return NotImplemented


def DICTAVAILLOSSFUNS(option, is_mask_exclude= False, option2_combine= None):
    if option2_combine:
        metrics_sub1 = DICTAVAILMETRICLASS(option, is_mask_exclude)
        metrics_sub2 = DICTAVAILMETRICLASS(option2_combine, is_mask_exclude)
        return CombineLossTwoMetrics(metrics_sub1, metrics_sub2, is_mask_exclude= is_mask_exclude)
    else:
        return DICTAVAILMETRICLASS(option, is_mask_exclude)


def DICTAVAILMETRICFUNS(option, is_mask_exclude=False):
    return DICTAVAILMETRICLASS(option, is_mask_exclude)