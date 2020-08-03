
from typing import Tuple, Callable
import numpy as np

from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
from tensorflow.keras import backend as K
import tensorflow as tf

from networks.metrics import Metrics as MetricsBase

_EPS = K._epsilon()
_SMOOTH = 1.0


class Metrics(MetricsBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(Metrics, self).__init__(is_mask_exclude)

    def compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        if self._is_mask_exclude:
            return self._compute_masked(K.flatten(y_true), K.flatten(y_pred))
        else:
            return self._compute(K.flatten(y_true), K.flatten(y_pred))

    def loss(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return self.compute(y_true, y_pred)

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        raise NotImplementedError

    def _compute_masked(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return self._compute(self._get_masked_input(y_true, y_true),
                             self._get_masked_input(y_pred, y_true))

    def _get_mask(self, y_true: K.Tensor) -> K.Tensor:
        return tf.where(K.equal(y_true, self._value_mask_exclude), K.zeros_like(y_true), K.ones_like(y_true))

    def _get_masked_input(self, y_input: K.Tensor, y_true: K.Tensor) -> K.Tensor:
        return tf.where(K.equal(y_true, self._value_mask_exclude), K.zeros_like(y_input), y_input)

    def get_renamed_compute(self) -> Callable:
        if self._name_func_out:
            return getattr(self, self._name_func_out)


class MetricsWithUncertainty(Metrics):
    # Composed uncertainty loss (ask Shuai)
    _epsilon_default = 0.01

    def __init__(self, metrics_loss: Metrics, epsilon: float = _epsilon_default) -> None:
        self._metrics_loss = metrics_loss
        self._epsilon = epsilon
        super(MetricsWithUncertainty, self).__init__(self._metrics_loss._is_mask_exclude)
        self._name_func_out = self._metrics_loss._name_func_out + '_uncertain'

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return (1.0 - self._epsilon) * self._metrics_loss._compute(y_true, y_pred) + \
               self._epsilon * self._metrics_loss._compute(K.ones_like(y_pred) / 3, y_pred)

    def _compute_masked(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return (1.0 - self._epsilon) * self._metrics_loss._compute_masked(y_true, y_pred) + \
               self._epsilon * self._metrics_loss._compute_masked(K.ones_like(y_pred) / 3, y_pred)


class CombineTwoMetrics(Metrics):

    def __init__(self, metrics_1: Metrics, metrics_2: Metrics, weights_metrics: Tuple[float, float] = (1.0, 1.0)) -> None:
        super(CombineTwoMetrics, self).__init__(False)
        self._metrics_1 = metrics_1
        self._metrics_2 = metrics_2
        self._weights_metrics = weights_metrics
        self._name_func_out = '_'.join(['combi', metrics_1._name_func_out, metrics_2._name_func_out])

    def compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return self._weights_metrics[0] * self._metrics_1.compute(y_true, y_pred) + \
               self._weights_metrics[1] * self._metrics_2.compute(y_true, y_pred)

    def loss(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return self._weights_metrics[0] * self._metrics_1.loss(y_true, y_pred) + \
               self._weights_metrics[1] * self._metrics_2.loss(y_true, y_pred)


class MeanSquaredError(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MeanSquaredError, self).__init__(is_mask_exclude)
        self._name_func_out  = 'mean_squared'

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return K.mean(K.square(y_pred - y_true), axis=-1)

    def _compute_masked(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        mask = self._get_mask(y_true)
        return K.mean(K.square(y_pred - y_true) * mask, axis=-1)


class MeanSquaredErrorLogarithmic(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MeanSquaredErrorLogarithmic, self).__init__(is_mask_exclude)
        self._name_func_out  = 'mean_squared_log'

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return K.mean(K.square(K.log(K.clip(y_pred, _EPS, None) + 1.0) -
                               K.log(K.clip(y_true, _EPS, None) + 1.0)), axis=-1)

    def _compute_masked(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        mask = self._get_mask(y_true)
        return K.mean(K.square(K.log(K.clip(y_pred, _EPS, None) + 1.0) -
                               K.log(K.clip(y_true, _EPS, None) + 1.0)) * mask, axis=-1)


class BinaryCrossEntropy(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(BinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_func_out = 'bin_cross'

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        #return K.mean(- y_true * K.log(y_pred + _EPS)
        #              - (1.0 - y_true) * K.log(1.0 - y_pred + _EPS))

    def _compute_masked(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        mask = self._get_mask(y_true)
        return K.mean(K.binary_crossentropy(y_true, y_pred) * mask, axis=-1)
        #return K.mean((- y_true * K.log(y_pred + _EPS)
        #               - (1.0 - y_true) * K.log(1.0 - y_pred + _EPS)) * mask)


class WeightedBinaryCrossEntropy(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(WeightedBinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_func_out = 'weight_bin_cross'

    def _get_weights(self, y_true: K.Tensor) -> Tuple[K.float, K.float]:
        num_class_1 = tf.count_nonzero(tf.where(K.equal(y_true, 1.0), K.ones_like(y_true), K.zeros_like(y_true)), dtype=K.int32)
        num_class_0 = tf.count_nonzero(tf.where(K.equal(y_true, 0.0), K.ones_like(y_true), K.zeros_like(y_true)), dtype=K.int32)
        return (1.0, K.cast(num_class_0, dtype=K.float32) / (K.cast(num_class_1, dtype=K.float32) + K.variable(_EPS)))

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        weights = self._get_weights(y_true)
        return K.mean(- weights[1] * y_true * K.log(y_pred + _EPS)
                      - weights[0] * (1.0 - y_true) * K.log(1.0 - y_pred + _EPS))

    def _compute_masked(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        weights = self._get_weights(y_true)
        mask = self._get_mask(y_true)
        return K.mean((- weights[1] * y_true * K.log(y_pred + _EPS)
                       - weights[0] * (1.0 - y_true) * K.log(1.0 - y_pred + _EPS)) * mask)


class WeightedBinaryCrossEntropyFixedWeights(Metrics):
    weights_no_masks_exclude = (1.0, 80.0)
    weights_mask_exclude = (1.0, 300.0)  # for LUVAR data
    #weights_mask_exclude = (1.0, 361.0)  # for DLCST data

    def __init__(self, is_mask_exclude: bool = False) -> None:
        if is_mask_exclude:
            self._weights = self.weights_mask_exclude
        else:
            self._weights = self.weights_no_masks_exclude
        super(WeightedBinaryCrossEntropyFixedWeights, self).__init__(is_mask_exclude)
        self._name_func_out = 'weight_bin_cross_fixed'

    def _get_weights(self, y_true: K.Tensor) -> Tuple[K.float, K.float]:
        return self._weights


class BinaryCrossEntropyFocalLoss(Metrics):
    # Binary cross entropy + Focal loss
    _gamma_default = 2.0

    def __init__(self, gamma: float = _gamma_default, is_mask_exclude: bool = False) -> None:
        self._gamma = gamma
        super(BinaryCrossEntropyFocalLoss, self).__init__(is_mask_exclude)
        self._name_func_out = 'bin_cross_focal_loss'

    def get_predprobs_classes(self, y_true: K.Tensor, y_pred: K.Tensor) -> Tuple[K.Tensor, K.Tensor]:
        prob_1 = tf.where(K.equal(y_true, 1.0), y_pred, K.ones_like(y_pred))
        prob_0 = tf.where(K.equal(y_true, 0.0), y_pred, K.zeros_like(y_pred))
        return (prob_1, prob_0)

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return K.mean(- y_true * K.pow(1.0 - y_pred, self._gamma) * K.log(y_pred + _EPS)
                      - (1.0 - y_true) * K.pow(y_pred, self._gamma) * K.log(1.0 - y_pred + _EPS))

    def _compute_masked(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        mask = self._get_mask(y_true)
        return K.mean((- y_true * K.pow(1.0 - y_pred, self._gamma) * K.log(y_pred + _EPS)
                       - (1.0 - y_true) * K.pow(y_pred, self._gamma) * K.log(1.0 - y_pred + _EPS)) * mask)


class DiceCoefficient(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(DiceCoefficient, self).__init__(is_mask_exclude)
        self._name_func_out = 'dice'

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return (2.0 * K.sum(y_true * y_pred)) / (K.sum(y_true) + K.sum(y_pred) + _SMOOTH)

    def loss(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return 1.0 - self.compute(y_true, y_pred)


class TruePositiveRate(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(TruePositiveRate, self).__init__(is_mask_exclude)
        self._name_func_out = 'tpr'

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return K.sum(y_true * y_pred) / (K.sum(y_true) + _SMOOTH)

    def loss(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return 1.0 - self.compute(y_true, y_pred)


class TrueNegativeRate(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(TrueNegativeRate, self).__init__(is_mask_exclude)
        self._name_func_out = 'tnr'

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return K.sum((1.0 - y_true) * (1.0 - y_pred)) / (K.sum((1.0 - y_true)) + _SMOOTH)

    def loss(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return 1.0 - self.compute(y_true, y_pred)


class FalsePositiveRate(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(FalsePositiveRate, self).__init__(is_mask_exclude)
        self._name_func_out = 'fpr'

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return K.sum((1.0 - y_true) * y_pred) / (K.sum((1.0 - y_true)) + _SMOOTH)


class FalseNegativeRate(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(FalseNegativeRate, self).__init__(is_mask_exclude)
        self._name_func_out = 'fnr'

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return K.sum(y_true * (1.0 - y_pred)) / (K.sum(y_true) + _SMOOTH)


class AirwayCompleteness(Metrics):
    _is_use_ytrue_cenlines = True
    _is_use_ypred_cenlines = False

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayCompleteness, self).__init__(is_mask_exclude)
        self._name_func_out = 'completeness'

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return K.sum(y_true * y_pred) / (K.sum(y_true) + _SMOOTH)


class AirwayVolumeLeakage(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayVolumeLeakage, self).__init__(is_mask_exclude)
        self._name_func_out = 'vol_leakage'

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return K.sum((1.0 - y_true) * y_pred) / (K.sum(y_pred) + _SMOOTH)


class AirwayCentrelineLeakage(Metrics):
    _is_use_ytrue_cenlines = False
    _is_use_ypred_cenlines = True

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayCentrelineLeakage, self).__init__(is_mask_exclude)
        self._name_func_out = 'cenline_leakage'

    def _compute(self, y_true: K.Tensor, y_pred: K.Tensor) -> K.Tensor:
        return K.sum((1.0 - y_true) * y_pred) / (K.sum(y_pred) + _SMOOTH)


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