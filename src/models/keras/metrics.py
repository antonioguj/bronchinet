
from typing import Tuple, Callable, Union

from tensorflow.keras import backend as K
import tensorflow as tf

from models.metrics import MetricBase

_EPS = K.epsilon()
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


class Metric(MetricBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(Metric, self).__init__(is_mask_exclude)

    def compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        if self._is_mask_exclude:
            return self._compute_masked(K.flatten(target), K.flatten(input))
        else:
            return self._compute(K.flatten(target), K.flatten(input))

    def lossfun(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return self.compute(target, input)

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return self._compute(self._get_masked_input(target, target),
                             self._get_masked_input(input, target))

    def _get_mask(self, target: tf.Tensor) -> tf.Tensor:
        return tf.where(K.equal(target, self._value_mask_exclude), K.zeros_like(target), K.ones_like(target))

    def _get_masked_input(self, input: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        return tf.where(K.equal(target, self._value_mask_exclude), K.zeros_like(input), input)

    def renamed_lossfun_backward_compat(self) -> Callable:
        setattr(self, 'loss', self.lossfun)
        out_fun_renamed = getattr(self, 'loss')
        out_fun_renamed.__func__.__name__ = 'loss'
        return out_fun_renamed

    def renamed_compute(self) -> Union[Callable, None]:
        if self._name_fun_out:
            setattr(self, self._name_fun_out, self.compute)
            out_fun_renamed = getattr(self, self._name_fun_out)
            out_fun_renamed.__func__.__name__ = self._name_fun_out
            return out_fun_renamed
        else:
            return None


class MetricWithUncertainty(Metric):
    # Composed uncertainty loss
    _epsilon_default = 0.01

    def __init__(self, metrics_loss: Metric, epsilon: float = _epsilon_default) -> None:
        self._metrics_loss = metrics_loss
        self._epsilon = epsilon
        super(MetricWithUncertainty, self).__init__(self._metrics_loss._is_mask_exclude)
        self._name_fun_out = self._metrics_loss._name_fun_out + '_uncertain'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return (1.0 - self._epsilon) * self._metrics_loss._compute(target, input) \
            + self._epsilon * self._metrics_loss._compute(K.ones_like(input) / 3, input)

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return (1.0 - self._epsilon) * self._metrics_loss._compute_masked(target, input) \
            + self._epsilon * self._metrics_loss._compute_masked(K.ones_like(input) / 3, input)


class CombineTwoMetrics(Metric):

    def __init__(self, metrics_1: Metric, metrics_2: Metric, weight_metric2over1: float = 1.0) -> None:
        super(CombineTwoMetrics, self).__init__(False)
        self._metrics_1 = metrics_1
        self._metrics_2 = metrics_2
        self._weight_metric2over1 = weight_metric2over1
        self._name_fun_out = '_'.join(['combi', metrics_1._name_fun_out, metrics_2._name_fun_out])

    def compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return self._metrics_1.compute(target, input) \
            + self._weight_metric2over1 * self._metrics_2.compute(target, input)

    def lossfun(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return self._metrics_1.lossfun(target, input) \
            + self._weight_metric2over1 * self._metrics_2.lossfun(target, input)


class MeanSquaredError(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MeanSquaredError, self).__init__(is_mask_exclude)
        self._name_fun_out = 'mean_squared'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return K.mean(K.square(input - target))

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        mask = self._get_mask(target)
        return K.mean(K.square(input - target) * mask)


class MeanSquaredErrorLogarithmic(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MeanSquaredErrorLogarithmic, self).__init__(is_mask_exclude)
        self._name_fun_out = 'mean_squared_log'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return K.mean(K.square(K.log(K.clip(input, _EPS, None) + 1.0)
                               - K.log(K.clip(target, _EPS, None) + 1.0)))

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        mask = self._get_mask(target)
        return K.mean(K.square(K.log(K.clip(input, _EPS, None) + 1.0)
                               - K.log(K.clip(target, _EPS, None) + 1.0)) * mask)


class BinaryCrossEntropy(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(BinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_fun_out = 'bin_cross'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return K.mean(K.binary_crossentropy(target, input))
        # return K.mean(- target * K.log(input + _EPS)
        #               - (1.0 - target) * K.log(1.0 - input + _EPS))

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        mask = self._get_mask(target)
        return K.mean(K.binary_crossentropy(target, input) * mask)
        # return K.mean((- target * K.log(input + _EPS)
        #                - (1.0 - target) * K.log(1.0 - input + _EPS)) * mask)


class WeightedBinaryCrossEntropy(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(WeightedBinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_fun_out = 'weight_bin_cross'

    def _get_weights(self, target: tf.Tensor) -> Tuple[float, float]:
        num_class_1 = tf.count_nonzero(tf.where(K.equal(target, 1.0), K.ones_like(target), K.zeros_like(target)),
                                       dtype=tf.int32)
        num_class_0 = tf.count_nonzero(tf.where(K.equal(target, 0.0), K.ones_like(target), K.zeros_like(target)),
                                       dtype=tf.int32)
        return (1.0, K.cast(num_class_0, dtype=tf.float32)
                / (K.cast(num_class_1, dtype=tf.float32) + K.variable(_EPS)))

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        weights = self._get_weights(target)
        return K.mean(- weights[1] * target * K.log(input + _EPS)
                      - weights[0] * (1.0 - target) * K.log(1.0 - input + _EPS))

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        weights = self._get_weights(target)
        mask = self._get_mask(target)
        return K.mean((- weights[1] * target * K.log(input + _EPS)
                       - weights[0] * (1.0 - target) * K.log(1.0 - input + _EPS)) * mask)


class WeightedBinaryCrossEntropyFixedWeights(WeightedBinaryCrossEntropy):
    weights_no_masks_exclude = (1.0, 80.0)
    weights_mask_exclude = (1.0, 300.0)  # for LUVAR data
    # weights_mask_exclude = (1.0, 361.0)  # for DLCST data

    def __init__(self, is_mask_exclude: bool = False) -> None:
        if is_mask_exclude:
            self._weights = self.weights_mask_exclude
        else:
            self._weights = self.weights_no_masks_exclude
        super(WeightedBinaryCrossEntropyFixedWeights, self).__init__(is_mask_exclude)
        self._name_fun_out = 'weight_bin_cross_fixed'

    def _get_weights(self, target: tf.Tensor) -> Tuple[float, float]:
        return self._weights


class BinaryCrossEntropyFocalLoss(Metric):
    # Binary cross entropy + Focal loss
    _gamma_default = 2.0

    def __init__(self, gamma: float = _gamma_default, is_mask_exclude: bool = False) -> None:
        self._gamma = gamma
        super(BinaryCrossEntropyFocalLoss, self).__init__(is_mask_exclude)
        self._name_fun_out = 'bin_cross_focal_loss'

    def get_predprobs_classes(self, target: tf.Tensor, input: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        prob_1 = tf.where(K.equal(target, 1.0), input, K.ones_like(input))
        prob_0 = tf.where(K.equal(target, 0.0), input, K.zeros_like(input))
        return (prob_1, prob_0)

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return K.mean(- target * K.pow(1.0 - input, self._gamma) * K.log(input + _EPS)
                      - (1.0 - target) * K.pow(input, self._gamma) * K.log(1.0 - input + _EPS))

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        mask = self._get_mask(target)
        return K.mean((- target * K.pow(1.0 - input, self._gamma) * K.log(input + _EPS)
                       - (1.0 - target) * K.pow(input, self._gamma) * K.log(1.0 - input + _EPS)) * mask)


class DiceCoefficient(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(DiceCoefficient, self).__init__(is_mask_exclude)
        self._name_fun_out = 'dice'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return (2.0 * K.sum(target * input)) / (K.sum(target) + K.sum(input) + _SMOOTH)

    def lossfun(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return 1.0 - self.compute(target, input)


class TruePositiveRate(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(TruePositiveRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'tp_rate'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return K.sum(target * input) / (K.sum(target) + _SMOOTH)

    def lossfun(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return 1.0 - self.compute(target, input)


class TrueNegativeRate(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(TrueNegativeRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'tn_rate'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return K.sum((1.0 - target) * (1.0 - input)) / (K.sum((1.0 - target)) + _SMOOTH)

    def lossfun(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return 1.0 - self.compute(target, input)


class FalsePositiveRate(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(FalsePositiveRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'fp_rate'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return K.sum((1.0 - target) * input) / (K.sum((1.0 - target)) + _SMOOTH)


class FalseNegativeRate(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(FalseNegativeRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'fn_rate'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return K.sum(target * (1.0 - input)) / (K.sum(target) + _SMOOTH)
