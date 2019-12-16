#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from tensorflow.python.keras.losses import mean_squared_error, binary_crossentropy
from tensorflow.python.keras import backend as K
import tensorflow as tf
from Common.ErrorMessages import *
import numpy as np

_eps = K.epsilon()
_smooth = 1.0


# VARIOUS METRICS:
class Metrics(object):
    max_size_memory_safe = 5e+08
    val_exclude = -1
    count = 0
    _isUse_reference_clines = False
    _isUse_predicted_clines = False

    def __init__(self, is_masks_exclude=False):
        self.is_masks_exclude = is_masks_exclude
        self.name_fun_out = None

    def loss(self, y_true, y_pred):
        return NotImplemented

    def compute_vec(self, y_true, y_pred):
        return NotImplemented

    def compute_vec_np(self, y_true, y_pred):
        return NotImplemented

    def compute_vec_masked(self, y_true, y_pred):
        return self.compute_vec(self.get_masked_array(y_true, y_true),
                                self.get_masked_array(y_true, y_pred))

    def compute_vec_masked_np(self, y_true, y_pred):
        return self.compute_vec_np(self.get_masked_array_np(y_true, y_true),
                                   self.get_masked_array_np(y_true, y_pred))

    def compute(self, y_true, y_pred):
        if self.is_masks_exclude:
            return self.compute_vec_masked(K.flatten(y_true), K.flatten(y_pred))
        else:
            return self.compute_vec(K.flatten(y_true), K.flatten(y_pred))

    def compute_np(self, y_true, y_pred):
        if self.is_masks_exclude:
            return self.compute_vec_masked_np(y_true.flatten(), y_pred.flatten())
        else:
            return self.compute_vec_np(y_true.flatten(), y_pred.flatten())

    def compute_np_safememory(self, y_true, y_pred):
        if(y_true.size > self.max_size_memory_safe):
            #if arrays are too large, split then in two and compute metrics twice, and return size-weighted metrics
            totaldim_0= y_true.shape[0]
            metrics_1 = self.compute_np(y_true[0:totaldim_0/2], y_pred[:totaldim_0/2])
            metrics_2 = self.compute_np(y_true[totaldim_0/2:],  y_pred[totaldim_0/2:])
            size_1    = y_true[0:totaldim_0/2].size
            size_2    = y_true[totaldim_0/2:].size
            return (metrics_1*size_1 + metrics_2*size_2)/(size_1 + size_2)
        else:
            return self.compute_np(y_true, y_pred)

    def get_mask(self, y_true):
        return tf.where(K.equal(y_true, self.val_exclude), K.zeros_like(y_true), K.ones_like(y_true))

    def get_masked_array(self, y_true, y_array):
        return tf.where(K.equal(y_true, self.val_exclude), K.zeros_like(y_array), y_array)

    def get_mask_np(self, y_true):
        return tf.where(y_true == self.val_exclude, 0, 1)

    def get_masked_array_np(self, y_true, y_array):
        return tf.where(y_true == self.val_exclude, 0, y_array)

    def get_renamed_compute(self):
        if self.name_fun_out:
            return getattr(self, self.name_fun_out)


# composed uncertainty loss (ask Shuai)
class MetricsWithUncertaintyLoss(Metrics):
    epsilon_default = 0.01

    def __init__(self, metrics_loss, epsilon=epsilon_default):
        self.metrics_loss = metrics_loss
        self.epsilon = epsilon
        super(MetricsWithUncertaintyLoss, self).__init__(self.metrics_loss.is_masks_exclude)
        #self.name_fun_out = 'bin_cross'
        self.name_fun_out = self.metrics_loss.name_fun_out + '_uncerloss'

    def compute_vec(self, y_true, y_pred):
        return (1.0 - self.epsilon) * self.metrics_loss.compute_vec(y_true, y_pred) + \
               self.epsilon * self.metrics_loss.compute_vec(K.ones_like(y_pred) / 3, y_pred)

    def compute_vec_masked(self, y_true, y_pred):
        return (1.0 - self.epsilon) * self.metrics_loss.compute_vec_masked(y_true, y_pred) + \
               self.epsilon * self.metrics_loss.compute_vec_masked(K.ones_like(y_pred) / 3, y_pred)

    def loss(self, y_true, y_pred):
        return (1.0 - self.epsilon) * self.metrics_loss.loss(y_true, y_pred) + \
               self.epsilon * self.metrics_loss.loss(K.ones_like(y_pred) / 3, y_pred)




# mean squared error
class MeanSquared(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(MeanSquared, self).__init__(is_masks_exclude)
        self.name_fun_out  = 'mean_squared'

    def compute_vec(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

    def compute_vec_masked(self, y_true, y_pred):
        mask = self.get_mask(y_true)
        return K.mean(K.square(y_pred - y_true) * mask, axis=-1)

    def compute_vec_np(self, y_true, y_pred):
        return np.mean(np.square(y_pred - y_true))

    def compute_vec_masked_np(self, y_true, y_pred):
        mask = self.get_mask_np(y_true)
        return np.mean(np.square(y_pred - y_true) * mask)

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def mean_squared(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# mean squared logarithmic error
class MeanSquaredLogarithmic(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(MeanSquaredLogarithmic, self).__init__(is_masks_exclude)
        self.name_fun_out  = 'mean_squared_logarithmic'

    def compute_vec(self, y_true, y_pred):
        return K.mean(K.square(K.log(K.clip(y_pred, _eps, None) + 1.0) -
                               K.log(K.clip(y_true, _eps, None) + 1.0)), axis=-1)

    def compute_vec_masked(self, y_true, y_pred):
        mask = self.get_mask(y_true)
        return K.mean(K.square(K.log(K.clip(y_pred, _eps, None) + 1.0) -
                               K.log(K.clip(y_true, _eps, None) + 1.0)) * mask, axis=-1)

    def compute_vec_np(self, y_true, y_pred):
        return np.mean(np.square(np.log(np.clip(y_pred, _eps, None) + 1.0) -
                                 np.log(np.clip(y_true, _eps, None) + 1.0)))

    def compute_vec_masked_np(self, y_true, y_pred):
        mask = self.get_mask_np(y_true)
        return np.mean(np.square(np.log(np.clip(y_pred, _eps, None) + 1.0) -
                                 np.log(np.clip(y_true, _eps, None) + 1.0)) * mask)

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def mean_squared(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# mean squared error
class MeanSquared_Tailored(MeanSquared):

    def __init__(self, is_masks_exclude=False, exp_y_true=1):
        super(MeanSquared_Tailored, self).__init__(is_masks_exclude)
        self.exp_y_true = exp_y_true

    def compute_vec(self, y_true, y_pred):
        #return K.mean(K.square(y_pred - y_true**self.exp_y_true), axis=-1)
        return K.mean(y_true * K.square(y_pred - y_true), axis=-1)

    def compute_vec_masked(self, y_true, y_pred):
        mask = self.get_mask(y_true)
        #return K.mean(K.square(y_pred - y_true**self.exp_y_true) * mask, axis=-1)
        return K.mean(y_true * K.square(y_pred - y_true) * mask, axis=-1)

    def compute_vec_np(self, y_true, y_pred):
        #return np.mean(np.square(y_pred - y_true**self.exp_y_true))
        return np.mean(y_true * np.square(y_pred - y_true))

    def compute_vec_masked_np(self, y_true, y_pred):
        mask = self.get_mask_np(y_true)
        #return np.mean(np.square(y_pred - y_true**self.exp_y_true) * mask)
        return np.mean(y_true * np.square(y_pred - y_true) * mask)


# binary cross entropy
class BinaryCrossEntropy(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(BinaryCrossEntropy, self).__init__(is_masks_exclude)
        self.name_fun_out = 'bin_cross'

    def compute_vec(self, y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        #return K.mean(- y_true * K.log(y_pred +_eps) -
        #              (1.0 - y_true) * K.log(1.0 - y_pred +_eps))

    def compute_vec_masked(self, y_true, y_pred):
        mask = self.get_mask(y_true)
        return K.mean(K.binary_crossentropy(y_true, y_pred) * mask, axis=-1)
        #return K.mean((- y_true * K.log(y_pred +_eps) -
        #               (1.0 - y_true) * K.log(1.0 - y_pred +_eps)) * mask)

    def compute_vec_np(self, y_true, y_pred):
        return np.mean(- y_true * np.log(y_pred +_eps) -
                       (1.0 - y_true) * np.log(1.0 - y_pred +_eps))

    def compute_vec_masked_np(self, y_true, y_pred):
        mask = self.get_mask_np(y_true)
        return np.mean((- y_true * np.log(y_pred +_eps) -
                        (1.0 - y_true) * np.log(1.0 - y_pred +_eps)) * mask)

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def bin_cross(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# weighted binary cross entropy
class WeightedBinaryCrossEntropy(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(WeightedBinaryCrossEntropy, self).__init__(is_masks_exclude)
        self.name_fun_out = 'wei_bin_cross'

    def get_weights(self, y_true):
        num_class_1 = tf.count_nonzero(tf.where(K.equal(y_true, 1.0),
                                                K.ones_like(y_true),
                                                K.zeros_like(y_true)), dtype=K.int32)
        num_class_0 = tf.count_nonzero(tf.where(K.equal(y_true, 0.0),
                                                K.ones_like(y_true),
                                                K.zeros_like(y_true)), dtype=K.int32)
        return (1.0, K.cast(num_class_0, dtype=K.float32) / (K.cast(num_class_1, dtype=K.float32) + K.variable(_eps)))

    def get_weights_np(self, y_true):
        num_class_1 = np.count_nonzero(y_true == 1)
        num_class_0 = np.count_nonzero(y_true == 0)
        return (1.0, num_class_0 / (float(num_class_1) +_eps))

    def compute_vec(self, y_true, y_pred):
        weights = self.get_weights(y_true)
        return K.mean(- weights[1] * y_true * K.log(y_pred +_eps) -
                      weights[0] * (1.0 - y_true) * K.log(1.0 - y_pred +_eps))

    def compute_vec_masked(self, y_true, y_pred):
        weights = self.get_weights(y_true)
        mask = self.get_mask(y_true)
        return K.mean((- weights[1] * y_true * K.log(y_pred +_eps) -
                       weights[0] * (1.0 - y_true) * K.log(1.0 - y_pred +_eps)) * mask)

    def compute_vec_np(self, y_true, y_pred):
        weights = self.get_weights_np(y_true)
        return np.mean(- weights[1] * y_true * np.log(y_pred +_eps) -
                       weights[0] * (1.0 - y_true) * np.log(1.0 - y_pred +_eps))

    def compute_vec_masked_np(self, y_true, y_pred):
        weights = self.get_weights_np(y_true)
        mask = self.get_mask_np(y_true)
        return np.mean((- weights[1] * y_true * np.log(y_pred +_eps) -
                        weights[0] * (1.0 - y_true) * np.log(1.0 - y_pred +_eps)) * mask)

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def wei_bin_cross(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# weighted binary cross entropy
class WeightedBinaryCrossEntropyFixedWeights(Metrics):
    #weights_noMasksExclude = [1.0, 80.0]
    #weights_masksExclude = [1.0, 300.0]  # for LUVAR data
    weights_masksExclude = [1.0, 361.0]  # for DLCST data

    def __init__(self, is_masks_exclude=False):
        if is_masks_exclude:
            self.weights = self.weights_masksExclude
        else:
            self.weights = self.weights_noMasksExclude
        super(WeightedBinaryCrossEntropyFixedWeights, self).__init__(is_masks_exclude)
        self.name_fun_out = 'wei_bin_cross_fixed'

    def compute_vec(self, y_true, y_pred):
        return K.mean(- self.weights[1] * y_true * K.log(y_pred +_eps) -
                      self.weights[0] * (1.0 - y_true) * K.log(1.0 - y_pred +_eps))

    def compute_vec_masked(self, y_true, y_pred):
        mask = self.get_mask(y_true)
        return K.mean((- self.weights[1] * y_true * K.log(y_pred +_eps) -
                       self.weights[0] * (1.0 - y_true) * K.log(1.0 - y_pred +_eps)) * mask)

    def compute_vec_np(self, y_true, y_pred):
        return np.mean(- self.weights[1] * y_true * np.log(y_pred +_eps) -
                       self.weights[0] * (1.0 - y_true) * np.log(1.0 - y_pred +_eps))

    def compute_vec_masked_np(self, y_true, y_pred):
        mask = self.get_mask_np(y_true)
        return np.mean((- self.weights[1] * y_true * np.log(y_pred +_eps) -
                        self.weights[0] * (1.0 - y_true) * np.log(1.0 - y_pred +_eps)) * mask)

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def wei_bin_cross_fixed(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# binary cross entropy + Focal loss
class BinaryCrossEntropyFocalLoss(BinaryCrossEntropy):
    gamma_default = 2.0

    def __init__(self, gamma=gamma_default, is_masks_exclude=False):
        self.gamma = gamma
        super(BinaryCrossEntropyFocalLoss, self).__init__(is_masks_exclude)
        self.name_fun_out = 'bin_cross'
        #self.name_fun_out = 'bin_cross_focal_loss'

    def get_clip_ypred(self, y_pred):
        # improve the stability of the focal loss
        return K.clip(y_pred, self.eps_clip, 1.0-self.eps_clip)

    def get_predprobs_classes(self, y_true, y_pred):
        prob_1 = tf.where(K.equal(y_true, 1.0), y_pred, K.ones_like(y_pred))
        prob_0 = tf.where(K.equal(y_true, 0.0), y_pred, K.zeros_like(y_pred))
        return (prob_1, prob_0)

    def compute_vec(self, y_true, y_pred):
        return K.mean(- y_true * K.pow(1.0 - y_pred, self.gamma) * K.log(y_pred +_eps) -
                      (1.0 - y_true) * K.pow(y_pred, self.gamma) * K.log(1.0 - y_pred +_eps))

    def compute_vec_masked(self, y_true, y_pred):
        mask = self.get_mask(y_true)
        return K.mean((- y_true * K.pow(1.0 - y_pred, self.gamma) * K.log(y_pred +_eps) -
                       (1.0 - y_true) * K.pow(y_pred, self.gamma) * K.log(1.0 - y_pred +_eps)) * mask)

    def compute_vec_np(self, y_true, y_pred):
        return np.mean(- y_true * pow(1.0 - y_pred, self.gamma) * np.log(y_pred +_eps) -
                       (1.0 - y_true) * pow(y_pred, self.gamma) * np.log(1.0 - y_pred +_eps))

    def compute_vec_masked_np(self, y_true, y_pred):
        mask = self.get_mask_np(y_true)
        return np.mean((- y_true * pow(1.0 - y_pred, self.gamma) * np.log(y_pred +_eps) -
                        (1.0 - y_true) * pow(y_pred, self.gamma) * np.log(1.0 - y_pred +_eps)) * mask)

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def bin_cross_focal_loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# weighted binary cross entropy + Focal Loss
class WeightedBinaryCrossEntropyFocalLoss(WeightedBinaryCrossEntropy):
    gamma_default = 2.0
    eps_clip = 1.0e-12

    def __init__(self, gamma=gamma_default, is_masks_exclude=False):
        self.gamma = gamma
        super(WeightedBinaryCrossEntropyFocalLoss, self).__init__(is_masks_exclude)
        self.name_fun_out = 'wei_bin_cross'
        #self.name_fun_out = 'wei_bin_cross_focal_loss'

    def get_clip_ypred(self, y_pred):
        # improve the stability of the focal loss
        return K.clip(y_pred, self.eps_clip, 1.0-self.eps_clip)

    def get_predprobs_classes(self, y_true, y_pred):
        prob_1 = tf.where(K.equal(y_true, 1.0), y_pred, K.ones_like(y_pred))
        prob_0 = tf.where(K.equal(y_true, 0.0), y_pred, K.zeros_like(y_pred))
        return (prob_1, prob_0)

    def compute_vec(self, y_true, y_pred):
        weights = self.get_weights(y_true)
        y_pred = self.get_clip_ypred(y_pred)
        return K.mean(- weights[1] * y_true * K.pow(1.0 - y_pred, self.gamma) * K.log(y_pred +_eps) -
                      weights[0] * (1.0 - y_true) * K.pow(y_pred, self.gamma) * K.log(1.0 - y_pred +_eps))
        #(prob_1, prob_0) = self.get_predprobs_classes(y_true, y_pred)
        #return K.mean(- weights[1] * K.pow(1.0 - prob_1, self.gamma) * K.log(prob_1) -
        #              weights[0] * K.pow(prob_0, self.gamma) * K.log(1.0 - prob_0))

    def compute_vec_masked(self, y_true, y_pred):
        weights = self.get_weights(y_true)
        mask = self.get_mask(y_true)
        y_pred = self.get_clip_ypred(y_pred)
        return K.mean((- weights[1] * y_true * K.pow(1.0 - y_pred, self.gamma) * K.log(y_pred +_eps) -
                       weights[0] * (1.0 - y_true) * K.pow(y_pred, self.gamma) * K.log(1.0 - y_pred +_eps)) * mask)
        #(prob_1, prob_0) = self.get_predprobs_classes(y_true, y_pred)
        #return K.mean((- weights[1] * K.pow(1.0 - prob_1, self.gamma) * K.log(prob_1 +_eps) -
        #              weights[0] * K.pow(prob_0, self.gamma) * K.log(1.0 - prob_0 +_eps)) * mask)

    def compute_vec_np(self, y_true, y_pred):
        weights = self.get_weights_np(y_true)
        return np.mean(- weights[1] * y_true * pow(1.0 - y_pred, self.gamma) * np.log(y_pred +_eps) -
                       weights[0] * (1.0 - y_true) * pow(y_pred, self.gamma) * np.log(1.0 - y_pred +_eps))

    def compute_vec_masked_np(self, y_true, y_pred):
        weights = self.get_weights_np(y_true)
        mask = self.get_mask_np(y_true)
        return np.mean((- weights[1] * y_true * pow(1.0 - y_pred, self.gamma) * np.log(y_pred +_eps) -
                        weights[0] * (1.0 - y_true) * pow(y_pred, self.gamma) * np.log(1.0 - y_pred +_eps)) * mask)

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def wei_bin_cross_focal_loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# categorical cross entropy
class CategoricalCrossEntropy(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(CategoricalCrossEntropy, self).__init__(is_masks_exclude)
        self.name_fun_out = 'cat_cross'

    def compute_vec(self, y_true, y_pred):
        return K.mean(K.categorical_crossentropy(y_true, y_pred), axis=-1)

    def compute_vec_np(self, y_true, y_pred):
        pass
        #return np.mean(-y_true * np.log(y_pred +_eps) - (1.0 - y_true) * np.log(1.0 - y_pred +_eps))

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def cat_cross(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# Dice coefficient
class DiceCoefficient(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(DiceCoefficient, self).__init__(is_masks_exclude)
        self.name_fun_out = 'dice'

    def compute_vec(self, y_true, y_pred):
        return (2.0*K.sum(y_true * y_pred)) / (K.sum(y_true) + K.sum(y_pred) +_smooth)

    def compute_vec_np(self, y_true, y_pred):
        return (2.0*np.sum(y_true * y_pred)) / (np.sum(y_true) + np.sum(y_pred) +_smooth)

    def loss(self, y_true, y_pred):
        return 1.0 - self.compute(y_true, y_pred)

    def dice(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# true positive rate
class TruePositiveRate(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(TruePositiveRate, self).__init__(is_masks_exclude)
        self.name_fun_out = 'tpr'

    def compute_vec(self, y_true, y_pred):
        return K.sum(y_true * y_pred) / (K.sum(y_true) +_smooth)

    def compute_vec_np(self, y_true, y_pred):
        return np.sum(y_true * y_pred) / (np.sum(y_true) +_smooth)

    def loss(self, y_true, y_pred):
        return 1.0 - self.compute(y_true, y_pred)

    def tpr(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# true negative rate
class TrueNegativeRate(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(TrueNegativeRate, self).__init__(is_masks_exclude)
        self.name_fun_out = 'tnr'

    def compute_vec(self, y_true, y_pred):
        return K.sum((1.0 - y_true) * (1.0 - y_pred)) / (K.sum((1.0 - y_true)) +_smooth)

    def compute_vec_np(self, y_true, y_pred):
        return np.sum((1.0 - y_true) * (1.0 - y_pred)) / (np.sum((1.0 - y_true)) +_smooth)

    def loss(self, y_true, y_pred):
        return 1.0 - self.compute(y_true, y_pred)

    def tnr(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# false positive rate
class FalsePositiveRate(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(FalsePositiveRate, self).__init__(is_masks_exclude)
        self.name_fun_out = 'fpr'

    def compute_vec(self, y_true, y_pred):
        return K.sum((1.0 - y_true) * y_pred) / (K.sum((1.0 - y_true)) +_smooth)

    def compute_vec_np(self, y_true, y_pred):
        return np.sum((1.0 - y_true) * y_pred) / (np.sum((1.0 - y_true)) +_smooth)

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def fpr(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# false negative rate
class FalseNegativeRate(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(FalseNegativeRate, self).__init__(is_masks_exclude)
        self.name_fun_out = 'fnr'

    def compute_vec(self, y_true, y_pred):
        return K.sum(y_true * (1.0 - y_pred)) / (K.sum(y_true) +_smooth)

    def compute_vec_np(self, y_true, y_pred):
        return np.sum(y_true * (1.0 - y_pred)) / (np.sum(y_true) +_smooth)

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def fnr(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# airways completeness (percentage ground-truth centrelines found inside the predicted airways)
class AirwayCompleteness(Metrics):
    _isUse_reference_clines = True
    _isUse_predicted_clines = False

    def __init__(self, is_masks_exclude=False):
        super(AirwayCompleteness, self).__init__(is_masks_exclude)
        self.name_fun_out = 'completeness'

    def compute_vec(self, y_true, y_pred):
        return K.sum(y_true * y_pred) / (K.sum(y_true) +_smooth)

    def compute_vec_np(self, y_true, y_pred):
        return np.sum(y_true * y_pred) / (np.sum(y_true) +_smooth)


# airways volume leakage (percentage of voxels from predicted airways found outside the ground-truth airways)
class AirwayVolumeLeakage(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(AirwayVolumeLeakage, self).__init__(is_masks_exclude)
        self.name_fun_out = 'volume_leakage'

    def compute_vec(self, y_true, y_pred):
        return K.sum((1.0 - y_true) * y_pred) / (K.sum(y_pred) +_smooth)

    def compute_vec_np(self, y_true, y_pred):
        return np.sum((1.0 - y_true) * y_pred) / (np.sum(y_pred) +_smooth)



# combination of two metrics
class CombineLossTwoMetrics(Metrics):
    weights_metrics = [1.0, 3.0]

    def __init__(self, metrics1, metrics2, is_masks_exclude=False):
        super(CombineLossTwoMetrics, self).__init__(is_masks_exclude)
        self.metrics1 = metrics1
        self.metrics2 = metrics2
        self.name_fun_out = '_'.join(['comb', metrics1.name_fun_out, metrics2.name_fun_out])

    def loss(self, y_true, y_pred):
        return self.weights_metrics[0] * self.metrics1.loss(y_true, y_pred) + \
               self.weights_metrics[1] * self.metrics2.loss(y_true, y_pred)


# all available metrics
def DICTAVAILMETRICLASS(option,
                        is_masks_exclude=False):
    list_metric_avail = ['MeanSquared', 'MeanSquared_Tailored',
                         'BinaryCrossEntropy', 'BinaryCrossEntropyFocalLoss', 'BinaryCrossEntropyUncertaintyLoss',
                         'WeightedBinaryCrossEntropyFixedWeights', 'WeightedBinaryCrossEntropy', 'WeightedBinaryCrossEntropyFocalLoss'
                         'DiceCoefficient', 'DiceCoefficientUncertaintyLoss'
                         'TruePositiveRate', 'TrueNegativeRate', 'FalsePositiveRate', 'FalseNegativeRate',
                         'AirwayCompleteness', 'AirwayVolumeLeakage']

    if   (option == 'MeanSquared'):
        return MeanSquared(is_masks_exclude=is_masks_exclude)
    elif (option == 'MeanSquared_Tailored'):
        return MeanSquared_Tailored(is_masks_exclude=is_masks_exclude)
    elif (option == 'BinaryCrossEntropy'):
        return BinaryCrossEntropy(is_masks_exclude=is_masks_exclude)
    elif (option == 'BinaryCrossEntropyFocalLoss'):
        return BinaryCrossEntropyFocalLoss(is_masks_exclude=is_masks_exclude)
    elif (option == 'BinaryCrossEntropyUncertaintyLoss'):
        return MetricsWithUncertaintyLoss(BinaryCrossEntropy(is_masks_exclude=is_masks_exclude))
    elif (option == 'WeightedBinaryCrossEntropyFixedWeights'):
        return WeightedBinaryCrossEntropyFixedWeights(is_masks_exclude=is_masks_exclude)
    elif (option == 'WeightedBinaryCrossEntropy'):
        return WeightedBinaryCrossEntropy(is_masks_exclude=is_masks_exclude)
    elif (option == 'WeightedBinaryCrossEntropyFocalLoss'):
        return WeightedBinaryCrossEntropyFocalLoss(is_masks_exclude=is_masks_exclude)
    elif (option == 'DiceCoefficient'):
        return DiceCoefficient(is_masks_exclude=is_masks_exclude)
    elif (option == 'DiceCoefficientUncertaintyLoss'):
        return MetricsWithUncertaintyLoss(DiceCoefficient(is_masks_exclude=is_masks_exclude))
    elif (option == 'TruePositiveRate'):
        return TruePositiveRate(is_masks_exclude=is_masks_exclude)
    elif (option == 'TrueNegativeRate'):
        return TrueNegativeRate(is_masks_exclude=is_masks_exclude)
    elif (option == 'FalsePositiveRate'):
        return FalsePositiveRate(is_masks_exclude=is_masks_exclude)
    elif (option == 'FalseNegativeRate'):
        return FalseNegativeRate(is_masks_exclude=is_masks_exclude)
    elif (option == 'AirwayCompleteness'):
        return AirwayCompleteness(is_masks_exclude=is_masks_exclude)
    elif (option == 'AirwayVolumeLeakage'):
        return AirwayVolumeLeakage(is_masks_exclude=is_masks_exclude)
    else:
        message = 'Metric \'%s\' chosen not found. Metrics available: \'%s\'...' %(option, ', '.join(list_metric_avail))
        CatchErrorException(message)
        return NotImplemented


def DICTAVAILLOSSFUNS(option, is_masks_exclude=False, option2_combine=None):
    if option2_combine:
        metrics_sub1 = DICTAVAILMETRICLASS(option, is_masks_exclude)
        metrics_sub2 = DICTAVAILMETRICLASS(option2_combine, is_masks_exclude)
        return CombineLossTwoMetrics(metrics_sub1, metrics_sub2, is_masks_exclude=is_masks_exclude)
    else:
        return DICTAVAILMETRICLASS(option, is_masks_exclude)


def DICTAVAILMETRICFUNS(option, is_masks_exclude=False):
    return DICTAVAILMETRICLASS(option, is_masks_exclude)
