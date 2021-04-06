
from typing import List, Tuple, Callable

from tensorflow.keras import backend as K
import tensorflow as tf
# from skimage import measure
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
# from scipy import signal
# from skimage.metrics import structural_similarity as ssim
# from scipy import signal

from common.exceptionmanager import catch_error_exception
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
                      'L1',
                      'L2',
                      'DSSIM',
                      'MultiScaleSSIM',
                      'Perceptual',
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

    def renamed_compute(self) -> Callable:
        if self._name_fun_out:
            setattr(self, self._name_fun_out, self.compute)
            out_fun_renamed = getattr(self, self._name_fun_out)
            out_fun_renamed.__func__.__name__ = self._name_fun_out
            return out_fun_renamed
        else:
            None


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
        return K.mean(K.square(input - target), axis=-1)

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        mask = self._get_mask(target)
        return K.mean(K.square(input - target) * mask, axis=-1)


class MeanSquaredErrorLogarithmic(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MeanSquaredErrorLogarithmic, self).__init__(is_mask_exclude)
        self._name_fun_out = 'mean_squared_log'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return K.mean(K.square(K.log(K.clip(input, _EPS, None) + 1.0)
                               - K.log(K.clip(target, _EPS, None) + 1.0)), axis=-1)

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        mask = self._get_mask(target)
        return K.mean(K.square(K.log(K.clip(input, _EPS, None) + 1.0)
                               - K.log(K.clip(target, _EPS, None) + 1.0)) * mask, axis=-1)


class BinaryCrossEntropy(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(BinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_fun_out = 'bin_cross'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return K.mean(K.binary_crossentropy(target, input), axis=-1)
        # return K.mean(- target * K.log(input + _EPS)
        #               - (1.0 - target) * K.log(1.0 - input + _EPS))

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        mask = self._get_mask(target)
        return K.mean(K.binary_crossentropy(target, input) * mask, axis=-1)
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


class WeightedBinaryCrossEntropyFixedWeights(Metric):
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


# ******************** METRICS USED IN MRI_ENHANCE METHOD ********************

class MetricModified(Metric):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MetricModified, self).__init__(is_mask_exclude)

    def compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        if self._is_mask_exclude:
            return self._compute_masked(target, input)
        else:
            return self._compute(target, input)

    def lossfun(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return self.compute(target, input)

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def _get_mask(self, target: tf.Tensor) -> tf.Tensor:
        return tf.where(K.equal(target, self._value_mask_exclude), K.zeros_like(target), K.ones_like(target))

    def _get_masked_input(self, input: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        return tf.where(K.equal(target, self._value_mask_exclude), K.zeros_like(input), input)

    def _factor_normalize(self, target: tf.Tensor) -> float:
        return tf.constant(1.0)


class CombineTwoMetricsModified(MetricModified):

    def __init__(self, metrics_1: MetricModified, metrics_2: MetricModified, weight_metric2over1: float = 1.0) -> None:
        super(CombineTwoMetricsModified, self).__init__(False)
        self._metrics_1 = metrics_1
        self._metrics_2 = metrics_2
        self._weight_metric2over1 = weight_metric2over1
        self._name_fun_out = '_'.join(['combi', metrics_1._name_fun_out, metrics_2._name_fun_out])

    def _get_factor_2ndmetric(self, target: tf.Tensor) -> float:
        # return self._weight_metric2over1 * self._metrics_1._factor_normalize(target) \
        #        / self._metrics_2._factor_normalize(target)
        return self._weight_metric2over1

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        factor_2ndmetric = self._get_factor_2ndmetric(target)
        return self._metrics_1._compute(target, input) + factor_2ndmetric * self._metrics_2._compute(target, input)

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        factor_2ndmetric = self._get_factor_2ndmetric(target)
        return self._metrics_1._compute_masked(target, input) \
            + factor_2ndmetric * self._metrics_2._compute_masked(target, input)


class L1(MetricModified):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(L1, self).__init__(is_mask_exclude)
        self._name_fun_out = 'l1'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        target = K.flatten(target)
        input = K.flatten(input)
        return K.mean(K.abs(input - target), axis=-1)

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        target = K.flatten(target)
        input = K.flatten(input)
        mask = self._get_mask(target)
        return K.mean(K.abs(input - target) * mask, axis=-1)

    def _factor_normalize(self, target: tf.Tensor) -> float:
        return tf.reduce_max(target)


class L2(MetricModified):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(L2, self).__init__(is_mask_exclude)
        self._name_fun_out = 'l2'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        target = K.flatten(target)
        input = K.flatten(input)
        return K.mean(K.square(input - target), axis=-1)

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        target = K.flatten(target)
        input = K.flatten(input)
        mask = self._get_mask(target)
        return K.mean(K.square(input - target) * mask, axis=-1)

    def _factor_normalize(self, target: tf.Tensor) -> float:
        return tf.reduce_max(target)


# Structural Dissimilarity Index
class DSSIM(MetricModified):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(DSSIM, self).__init__(is_mask_exclude)
        self._name_fun_out = 'dssim'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        # return tf.reduce_mean(tf.image.ssim(target, input, 255.0))
        return tf.reduce_mean(tf.image.ssim(target, input, 255.0, filter_size=10))
        # return tf.reduce_mean(tf.image.ssim(target, input, max_val=255.0, filter_size=11, filter_sigma=1.5,
        #                                     k1=0.01, k2=0.03))
        # return measure.compare_ssim(target, input, multichannel=True, data_range=255.0)

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return self._compute(self._get_masked_input(target, target),
                             self._get_masked_input(input, target))

    def lossfun(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return (1.0 - self.compute(target, input)) / 2.0


# MultiScale Structural Similarity Index
class MultiScaleSSIM(MetricModified):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MultiScaleSSIM, self).__init__(is_mask_exclude)
        self._name_fun_out = 'multiscalessim'

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.image.ssim_multiscale(target, input, max_val=255))
        # return multiscalessim(target, input)

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return self._compute(self._get_masked_input(target, target),
                             self._get_masked_input(input, target))

    def lossfun(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        return self.compute(target, input)


# Perceptual loss with pretrained VGG16 network
class Perceptual(MetricModified):
    _layers_vgg16_calcloss_default = ['block1_conv1', 'block2_conv1', 'block3_conv1']
    _weights_vgg16_calcloss_default = [0.65, 0.30, 0.05]

    def __init__(self, is_mask_exclude: bool = False,
                 size_image: Tuple[int, int] = (256, 256),
                 layers_vgg16_calcloss: List[str] = _layers_vgg16_calcloss_default,
                 weights_vgg16_calcloss: List[float] = _weights_vgg16_calcloss_default,
                 prop_reduce_insize_vgg16_calcloss: float = 0.0
                 ) -> None:
        super(Perceptual, self).__init__(is_mask_exclude)
        self._name_fun_out = 'perceptual'
        self._size_slice_image = size_image[-2:]
        self._layers_vgg16_calcloss = layers_vgg16_calcloss
        self._weights_vgg16_calcloss = weights_vgg16_calcloss
        if len(layers_vgg16_calcloss) == 1:
            self._is_calcloss_feats_alllayer = False
            self._weights_vgg16_calcloss = [1.0]
        else:
            self._is_calcloss_feats_alllayer = True

        # use smaller input to the VGG16 net, centered from the input image, to reduce memory footprint
        reduced_size_dim1_input_vgg16 = int(self._size_slice_image[0] * (1.0 - prop_reduce_insize_vgg16_calcloss))
        self._size_input_vgg16 = (reduced_size_dim1_input_vgg16, self._size_slice_image[1])
        self._input_vgg16_shape = (*self._size_input_vgg16, 3)
        self._model_vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=self._input_vgg16_shape)
        self._model_vgg16.trainable = False

        if len(size_image) == 2:
            self._preprocess_inarrays = self._preprocess_inarrays_2d
        elif len(size_image) == 3:
            self._preprocess_inarrays = self._preprocess_inarrays_3d
        else:
            message = 'Perceptual:__init__: wrong \'ndims\': %s...' % (len(size_image))
            catch_error_exception(message)

        # cropping bounding box to extract the smaller input image to the VGG16 net from the input image
        self._crop_boundbox = self._calc_boundbox_centered(self._size_slice_image, self._size_input_vgg16)

        # function to compute the feature maps at the spec. layers of VGG16 net when evaluated on the input image
        self._output_layers_vgg16_calcloss = [self._model_vgg16.get_layer(iname).output
                                              for iname in self._layers_vgg16_calcloss]
        self._func_get_featuremaps_vgg16 = Model(inputs=self._model_vgg16.input,
                                                 outputs=self._output_layers_vgg16_calcloss)

        if is_mask_exclude:
            # when applying the masking: get the size(s) of the VGG16 layer(s) where the loss is computed,
            # and the times pooling is applied to get there, to compute the mask with the correct size
            model_vgg16_names_layers = [ilayer.name for ilayer in self._model_vgg16.layers]

            self._list_num_pooling_until_layers_vgg16_calcloss = []
            for ilayer_name in self._layers_vgg16_calcloss:
                num_pooling_until_layer_this = 0
                for jname in model_vgg16_names_layers:
                    if 'pool' in jname:
                        num_pooling_until_layer_this += 1
                    if jname == ilayer_name:
                        break
                self._list_num_pooling_until_layers_vgg16_calcloss.append(num_pooling_until_layer_this)

        print('\nUse Pretrained VGG16 model to compute the loss...')
        # print(self._modelVGG16.summary())
        print('1: Evaluate the prediction and ground-truth on the VGG16 model...')
        print('2: Compute loss as the L2 norm of the difference of feature maps from VGG16 layers: %s\n'
              % (self._layers_vgg16_calcloss))

        self._counter_call_compute_vec = 0

    def _calc_boundbox_centered(self, in_size_boundbox: Tuple[int, int],
                                out_size_boundbox: Tuple[int, int]
                                ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        boundbox_origin = (int((in_size_boundbox[0] - out_size_boundbox[0]) / 2),
                           int((in_size_boundbox[1] - out_size_boundbox[1]) / 2))
        out_boundbox = ((boundbox_origin[0], boundbox_origin[0] + out_size_boundbox[0]),
                        (boundbox_origin[1], boundbox_origin[1] + out_size_boundbox[1]))
        return out_boundbox

    def _crop_image(self, in_image: tf.Tensor, in_boundbox: Tuple[Tuple[int, int], Tuple[int, int]]) -> tf.Tensor:
        return in_image[...,  # first dim for batch size, if needed
                        in_boundbox[0][0]:in_boundbox[0][1],
                        in_boundbox[1][0]:in_boundbox[1][1],
                        :]      # last dim for channels

    def _factor_normalize(self, target: tf.Tensor) -> float:
        return tf.reduce_max(target)

    def _preprocess_inarrays_2d(self, target: tf.Tensor, input: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # crop the image to the input size to the VGG16 net
        target = self._crop_image(target, self._crop_boundbox)
        input = self._crop_image(input, self._crop_boundbox)

        # gray-scale image in RGB: repeat the array in 3 input channels
        target = tf.stack((tf.squeeze(target, axis=-1),) * 3, axis=-1)
        input = tf.stack((tf.squeeze(input, axis=-1),) * 3, axis=-1)

        return (target, input)

    def _preprocess_inarrays_3d(self, target: tf.Tensor, input: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # merge 1st dim (batch_size) and 2nd dim (img_dimZ) together
        y_new_shape = (-1, *input.shape[-3:])
        target = tf.reshape(target, y_new_shape)
        input = tf.reshape(input, y_new_shape)

        # crop the image to the input size to the VGG16 net
        target = self._crop_image(target, self._crop_boundbox)
        input = self._crop_image(input, self._crop_boundbox)

        # gray-scale image in RGB: repeat the array in 3 input channels
        target = tf.stack((tf.squeeze(target, axis=-1),) * 3, axis=-1)
        input = tf.stack((tf.squeeze(input, axis=-1),) * 3, axis=-1)

        return (target, input)

    def _get_mask_size_layer_vgg16(self, target: tf.Tensor, ilayer_vgg16_calcloss: int) -> tf.Tensor:
        for i in range(self._list_num_pooling_until_layers_vgg16_calcloss[ilayer_vgg16_calcloss]):
            target = tf.nn.max_pool2d(target, ksize=(2, 2), strides=(2, 2), padding='SAME')
        mask = self._get_mask(target[..., 0])
        return tf.expand_dims(mask, axis=-1)

    def _compute(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        if self._is_calcloss_feats_alllayer:
            return self._compute_feats_alllayers(target, input)
        else:
            return self._compute_feats_1layer(target, input)

    def _compute_masked(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        if self._is_calcloss_feats_alllayer:
            return self._compute_masked_feats_alllayers(target, input)
        else:
            return self._compute_masked_feats_1layer(target, input)

    def _compute_feats_1layer(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        # preprocess input arrays to VGG16 net
        (target, input) = self._preprocess_inarrays(target, input)

        # feature maps from VGG16 when evaluated on "target" / "input"
        featmaps_target = self._func_get_featuremaps_vgg16(target)
        featmaps_input = self._func_get_featuremaps_vgg16(input)
        featmaps_target = K.flatten(featmaps_target)
        featmaps_input = K.flatten(featmaps_input)

        # loss as the root mean squared error between the two feature maps
        return K.mean(K.square(featmaps_target - featmaps_input), axis=-1)

    def _compute_feats_alllayers(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        # preprocess input arrays to VGG16 net
        (target, input) = self._preprocess_inarrays(target, input)

        # feature maps from VGG16 when evaluated on "target" / "input"
        featmaps_target = self._func_get_featuremaps_vgg16(target)
        featmaps_input = self._func_get_featuremaps_vgg16(input)
        featmaps_target = [K.flatten(feats_ilayer) for feats_ilayer in featmaps_target]
        featmaps_input = [K.flatten(feats_ilayer) for feats_ilayer in featmaps_input]

        # list of root mean squared errors between the two feature maps on ALL layers
        msqrt_error_featmaps = [K.mean(K.square(feats_target_ilay - feats_input_ilay), axis=-1)
                                for (feats_target_ilay, feats_input_ilay) in zip(featmaps_target, featmaps_input)]
        return K.sum(tf.multiply(msqrt_error_featmaps, self._weights_vgg16_calcloss))

    def _compute_masked_feats_1layer(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        # preprocess input arrays to VGG16 net
        (target, input) = self._preprocess_inarrays(target, input)

        # mask with size of the chosen layer of VGG16
        mask = self._get_mask_size_layer_vgg16(target, 0)

        # feature maps from VGG16 when evaluated on "target" / "input"
        featmaps_target = self._func_get_featuremaps_vgg16(target)
        featmaps_input = self._func_get_featuremaps_vgg16(input)

        # loss as the root mean squared error between the two feature maps, with the squared error array being masked
        return K.mean(K.flatten(tf.multiply(K.square(featmaps_target - featmaps_input), mask)), axis=-1)

    def _compute_masked_feats_alllayers(self, target: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        # preprocess input arrays to VGG16 net
        (target, input) = self._preprocess_inarrays(target, input)

        # feature maps from VGG16 when evaluated on "target" / "input"
        featmaps_target = self._func_get_featuremaps_vgg16(target)
        featmaps_input = self._func_get_featuremaps_vgg16(input)

        # masks with sizes of ALL the chosen layers of VGG16
        masks = [self._get_mask_size_layer_vgg16(target, i) for i in range(len(self._layers_vgg16_calcloss))]

        # list of root mean squared errors between the two feature maps on ALL layers,
        # with the squared error array being masked
        msqrt_error_featmaps = \
            [K.mean(K.flatten(tf.multiply(K.square(feats_target_ilay - feats_input_ilay), mask_ilay)), axis=-1)
             for (feats_target_ilay, feats_input_ilay, mask_ilay) in zip(featmaps_target, featmaps_input, masks)]
        return K.sum(tf.multiply(msqrt_error_featmaps, self._weights_vgg16_calcloss))
