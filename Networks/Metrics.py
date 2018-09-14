#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from keras import backend as K
import numpy as np

_eps = 1.0e-12
_smooth = 1.0


class LossFunction(object):
    def __init__(self, Metrics):
        self.Metrics = Metrics

    def compute(self, y_true, y_pred):
        return self.Metrics.loss(y_true, y_pred)


# DIFFERENT METRICS:
class Metrics(object):

    max_size_memory_safe = 5e+08
    val_exclude = -1
    count = 0

    def __init__(self, isMasksExclude=False):
        self.isMasksExclude = isMasksExclude
        self.name_out = None

    def compute(self, y_true, y_pred):
        if self.isMasksExclude:
            return self.compute_vec_masked(K.flatten(y_true),
                                           K.flatten(y_pred))
        else:
            return self.compute_vec(K.flatten(y_true),
                                    K.flatten(y_pred))

    def compute_vec(self, y_true, y_pred):
        pass

    def compute_vec_masked(self, y_true, y_pred):
        pass

    def compute_np(self, y_true, y_pred):
        if self.isMasksExclude:
            return self.compute_vec_masked_np(y_true.flatten(),
                                              y_pred.flatten())
        else:
            return self.compute_vec_np(y_true.flatten(),
                                       y_pred.flatten())

    def compute_vec_np(self, y_true, y_pred):
        pass

    def compute_vec_masked_np(self, y_true, y_pred):
        pass

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
        return K.tf.where(K.tf.equal(y_true, self.val_exclude), K.zeros_like(y_true), K.ones_like(y_true))

    def get_masked_array(self, y_true, y_array):
        return K.tf.where(K.tf.equal(y_true, self.val_exclude), K.zeros_like(y_array), y_array)

    def get_mask_np(self, y_true):
        return np.where(y_true == self.val_exclude, 0, 1)

    def get_masked_array_np(self, y_true, y_array):
        return np.where(y_true == self.val_exclude, 0, y_array)

    def loss(self, y_true, y_pred):
        pass

    def get_renamed_compute(self):
        if self.name_out:
            return getattr(self, self.name_out)


# Binary Cross entropy
class BinaryCrossEntropy(Metrics):

    def __init__(self, isMasksExclude=False):
        super(BinaryCrossEntropy, self).__init__(isMasksExclude)
        self.name_out = 'bin_cross'

    def compute_vec(self, y_true, y_pred):
        #return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        return K.mean(- y_true * K.log(y_pred +_eps) -
                      (1.0 - y_true) * K.log(1.0 - y_pred +_eps))

    def compute_vec_masked(self, y_true, y_pred):
        mask = self.get_mask(y_true)
        #return K.mean(K.binary_crossentropy(y_true, y_pred) * mask, axis=-1)
        return K.mean((- y_true * K.log(y_pred +_eps) -
                       (1.0 - y_true) * K.log(1.0 - y_pred +_eps)) * mask)

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


# Binary Cross entropy + Focal loss
class BinaryCrossEntropyFocalLoss(BinaryCrossEntropy):

    param_gamma_default = 2.0

    def __init__(self, param_gamma=param_gamma_default, isMasksExclude=False):
        self.param_gamma = param_gamma
        super(BinaryCrossEntropyFocalLoss, self).__init__(isMasksExclude)
        self.name_out = 'bin_cross'
        #self.name_out = 'bin_cross_focal_loss'

    def get_clip_ypred(self, y_pred):
        # improve the stability of the focal loss
        return K.clip(y_pred, self.eps_clip, 1.0-self.eps_clip)

    def get_predprobs_classes(self, y_true, y_pred):
        prob_1 = K.tf.where(K.tf.equal(y_true, 1.0), y_pred, K.ones_like(y_pred))
        prob_0 = K.tf.where(K.tf.equal(y_true, 0.0), y_pred, K.zeros_like(y_pred))
        return (prob_1, prob_0)

    def compute_vec(self, y_true, y_pred):
        return K.mean(- y_true * K.pow(1.0 - y_pred, self.param_gamma) * K.log(y_pred +_eps) -
                      (1.0 - y_true) * K.pow(y_pred, self.param_gamma) * K.log(1.0 - y_pred +_eps))

    def compute_vec_masked(self, y_true, y_pred):
        mask = self.get_mask(y_true)
        return K.mean((- y_true * K.pow(1.0 - y_pred, self.param_gamma) * K.log(y_pred +_eps) -
                       (1.0 - y_true) * K.pow(y_pred, self.param_gamma) * K.log(1.0 - y_pred +_eps)) * mask)

    def compute_vec_np(self, y_true, y_pred):
        return np.mean(- y_true * pow(1.0 - y_pred, self.param_gamma) * np.log(y_pred +_eps) -
                       (1.0 - y_true) * pow(y_pred, self.param_gamma) * np.log(1.0 - y_pred +_eps))

    def compute_vec_masked_np(self, y_true, y_pred):
        mask = self.get_mask_np(y_true)
        return np.mean((- y_true * pow(1.0 - y_pred, self.param_gamma) * np.log(y_pred +_eps) -
                        (1.0 - y_true) * pow(y_pred, self.param_gamma) * np.log(1.0 - y_pred +_eps)) * mask)

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def bin_cross_focal_loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# Weighted Binary Cross entropy
class WeightedBinaryCrossEntropyFixedWeights(Metrics):
    #weights_noMasksExclude = [1.0, 80.0]
    #weights_masksExclude = [1.0, 300.0]  # for LUVAR data
    weights_masksExclude = [1.0, 361.0]  # for DLCST data

    def __init__(self, isMasksExclude=False):
        if isMasksExclude:
            self.weights = self.weights_masksExclude
        else:
            self.weights = self.weights_noMasksExclude
        super(WeightedBinaryCrossEntropyFixedWeights, self).__init__(isMasksExclude)
        self.name_out = 'wei_bin_cross_fixed'

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


# Weighted Binary Cross entropy
class WeightedBinaryCrossEntropy(Metrics):

    def __init__(self, isMasksExclude=False):
        super(WeightedBinaryCrossEntropy, self).__init__(isMasksExclude)
        self.name_out = 'wei_bin_cross'

    def get_weights(self, y_true):
        num_class_1 = K.tf.count_nonzero(K.tf.where(K.tf.equal(y_true, 1.0),
                                                    K.ones_like(y_true),
                                                    K.zeros_like(y_true)), dtype=K.tf.int32)
        num_class_0 = K.tf.count_nonzero(K.tf.where(K.tf.equal(y_true, 0.0),
                                                    K.ones_like(y_true),
                                                    K.zeros_like(y_true)), dtype=K.tf.int32)
        return (1.0, K.cast(num_class_0, dtype=K.tf.float32) / (K.cast(num_class_1, dtype=K.tf.float32) + K.variable(_eps)))

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


# Weighted Binary Cross entropy + Focal Loss
class WeightedBinaryCrossEntropyFocalLoss(WeightedBinaryCrossEntropy):

    param_gamma_default = 2.0
    eps_clip = 1.0e-12

    def __init__(self, param_gamma=param_gamma_default, isMasksExclude=False):
        self.param_gamma = param_gamma
        super(WeightedBinaryCrossEntropyFocalLoss, self).__init__(isMasksExclude)
        self.name_out = 'wei_bin_cross'
        #self.name_out = 'wei_bin_cross_focal_loss'

    def get_clip_ypred(self, y_pred):
        # improve the stability of the focal loss
        return K.clip(y_pred, self.eps_clip, 1.0-self.eps_clip)

    def get_predprobs_classes(self, y_true, y_pred):
        prob_1 = K.tf.where(K.tf.equal(y_true, 1.0), y_pred, K.ones_like(y_pred))
        prob_0 = K.tf.where(K.tf.equal(y_true, 0.0), y_pred, K.zeros_like(y_pred))
        return (prob_1, prob_0)

    def compute_vec(self, y_true, y_pred):
        weights = self.get_weights(y_true)
        y_pred = self.get_clip_ypred(y_pred)
        return K.mean(- weights[1] * y_true * K.pow(1.0 - y_pred, self.param_gamma) * K.log(y_pred +_eps) -
                      weights[0] * (1.0 - y_true) * K.pow(y_pred, self.param_gamma) * K.log(1.0 - y_pred +_eps))
        #(prob_1, prob_0) = self.get_predprobs_classes(y_true, y_pred)
        #return K.mean(- weights[1] * K.pow(1.0 - prob_1, self.param_gamma) * K.log(prob_1) -
        #              weights[0] * K.pow(prob_0, self.param_gamma) * K.log(1.0 - prob_0))

    def compute_vec_masked(self, y_true, y_pred):
        weights = self.get_weights(y_true)
        mask = self.get_mask(y_true)
        y_pred = self.get_clip_ypred(y_pred)
        return K.mean((- weights[1] * y_true * K.pow(1.0 - y_pred, self.param_gamma) * K.log(y_pred +_eps) -
                       weights[0] * (1.0 - y_true) * K.pow(y_pred, self.param_gamma) * K.log(1.0 - y_pred +_eps)) * mask)
        #(prob_1, prob_0) = self.get_predprobs_classes(y_true, y_pred)
        #return K.mean((- weights[1] * K.pow(1.0 - prob_1, self.param_gamma) * K.log(prob_1 +_eps) -
        #              weights[0] * K.pow(prob_0, self.param_gamma) * K.log(1.0 - prob_0 +_eps)) * mask)

    def compute_vec_np(self, y_true, y_pred):
        weights = self.get_weights_np(y_true)
        return np.mean(- weights[1] * y_true * pow(1.0 - y_pred, self.param_gamma) * np.log(y_pred +_eps) -
                       weights[0] * (1.0 - y_true) * pow(y_pred, self.param_gamma) * np.log(1.0 - y_pred +_eps))

    def compute_vec_masked_np(self, y_true, y_pred):
        weights = self.get_weights_np(y_true)
        mask = self.get_mask_np(y_true)
        return np.mean((- weights[1] * y_true * pow(1.0 - y_pred, self.param_gamma) * np.log(y_pred +_eps) -
                        weights[0] * (1.0 - y_true) * pow(y_pred, self.param_gamma) * np.log(1.0 - y_pred +_eps)) * mask)

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def wei_bin_cross_focal_loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# Categorical Cross entropy
class CategoricalCrossEntropy(Metrics):
    def __init__(self, isMasksExclude=False):
        super(CategoricalCrossEntropy, self).__init__(isMasksExclude)
        self.name_out = 'cat_cross'

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

    def __init__(self, isMasksExclude=False):
        super(DiceCoefficient, self).__init__(isMasksExclude)
        self.name_out = 'dice'

    def compute_vec(self, y_true, y_pred):
        return (2.0*K.sum(y_true * y_pred)) / (K.sum(y_true) + K.sum(y_pred) +_smooth)

    def compute_vec_masked(self, y_true, y_pred):
        return self.compute_vec(self.get_masked_array(y_true, y_true), self.get_masked_array(y_true, y_pred))

    def compute_vec_np(self, y_true, y_pred):
        return (2.0*np.sum(y_true * y_pred)) / (np.sum(y_true) + np.sum(y_pred) +_smooth)

    def compute_vec_masked_np(self, y_true, y_pred):
        return self.compute_vec_np(self.get_masked_array_np(y_true, y_true), self.get_masked_array_np(y_true, y_pred))

    def loss(self, y_true, y_pred):
        return 1.0-self.compute(y_true, y_pred)

    def dice(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# True positive rate
class TruePositiveRate(Metrics):

    def __init__(self, isMasksExclude=False):
        super(TruePositiveRate, self).__init__(isMasksExclude)
        self.name_out = 'tpr'

    def compute_vec(self, y_true, y_pred):
        return K.sum(y_true * y_pred) / (K.sum(y_true) +_smooth)

    def compute_vec_masked(self, y_true, y_pred):
        return self.compute_vec(self.get_masked_array(y_true, y_true), self.get_masked_array(y_true, y_pred))

    def compute_vec_np(self, y_true, y_pred):
        return np.sum(y_true * y_pred) / (np.sum(y_true) +_smooth)

    def compute_vec_masked_np(self, y_true, y_pred):
        return self.compute_vec_np(self.get_masked_array_np(y_true, y_true), self.get_masked_array_np(y_true, y_pred))

    def loss(self, y_true, y_pred):
        return 1.0-self.compute(y_true, y_pred)

    def tpr(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# True negative rate
class TrueNegativeRate(Metrics):

    def __init__(self, isMasksExclude=False):
        super(TrueNegativeRate, self).__init__(isMasksExclude)
        self.name_out = 'tnr'

    def compute_vec(self, y_true, y_pred):
        return K.sum((1.0 - y_true) * (1.0 - y_pred)) / (K.sum((1.0 - y_true)) +_smooth)

    def compute_vec_masked(self, y_true, y_pred):
        return self.compute_vec(self.get_masked_array(y_true, y_true), self.get_masked_array(y_true, y_pred))

    def compute_vec_np(self, y_true, y_pred):
        return np.sum((1.0 - y_true) * (1.0 - y_pred)) / (np.sum((1.0 - y_true)) +_smooth)

    def compute_vec_masked_np(self, y_true, y_pred):
        return self.compute_vec_np(self.get_masked_array_np(y_true, y_true), self.get_masked_array_np(y_true, y_pred))

    def loss(self, y_true, y_pred):
        return 1.0-self.compute(y_true, y_pred)

    def tnr(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# False positive rate
class FalsePositiveRate(Metrics):

    def __init__(self, isMasksExclude=False):
        super(FalsePositiveRate, self).__init__(isMasksExclude)
        self.name_out = 'fpr'

    def compute_vec(self, y_true, y_pred):
        return K.sum((1.0 - y_true) * y_pred) / (K.sum((1.0 - y_true)) +_smooth)

    def compute_vec_masked(self, y_true, y_pred):
        return self.compute_vec(self.get_masked_array(y_true, y_true), self.get_masked_array(y_true, y_pred))

    def compute_vec_np(self, y_true, y_pred):
        return np.sum((1.0 - y_true) * y_pred) / (np.sum((1.0 - y_true)) +_smooth)

    def compute_vec_masked_np(self, y_true, y_pred):
        return self.compute_vec_np(self.get_masked_array_np(y_true, y_true), self.get_masked_array_np(y_true, y_pred))

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def fpr(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# False negative rate
class FalseNegativeRate(Metrics):

    def __init__(self, isMasksExclude=False):
        super(FalseNegativeRate, self).__init__(isMasksExclude)
        self.name_out = 'fnr'

    def compute_vec(self, y_true, y_pred):
        return K.sum(y_true * (1.0 - y_pred)) / (K.sum(y_true) +_smooth)

    def compute_vec_masked(self, y_true, y_pred):
        return self.compute_vec(self.get_masked_array(y_true, y_true), self.get_masked_array(y_true, y_pred))

    def compute_vec_np(self, y_true, y_pred):
        return np.sum(y_true * (1.0 - y_pred)) / (np.sum(y_true) +_smooth)

    def compute_vec_masked_np(self, y_true, y_pred):
        return self.compute_vec_np(self.get_masked_array_np(y_true, y_true), self.get_masked_array_np(y_true, y_pred))

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def fnr(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# Airways completeness (percentage ground-truth centrelines found inside the predicted airways)
class AirwayCompleteness(Metrics):

    def __init__(self):
        super(AirwayCompleteness, self).__init__(isMasksExclude=False)
        self.name_out = 'completeness'

    def compute_vec(self, y_centreline, y_pred_airway):
        return K.sum(y_centreline * y_pred_airway) / (K.sum(y_centreline) +_smooth)

    def compute_vec_np(self, y_centreline, y_pred_airway):
        return np.sum(y_centreline * y_pred_airway) / (np.sum(y_centreline) +_smooth)


# Airways volume leakage (percentage of voxels from predicted airways found outside the ground-truth airways)
class AirwayVolumeLeakage(Metrics):

    def __init__(self):
        super(AirwayVolumeLeakage, self).__init__(isMasksExclude=False)
        self.name_out = 'volume_leakage'

    def compute_vec(self, y_true_airway, y_pred_airway):
        return K.sum((1.0 - y_true_airway) * y_pred_airway) / (K.sum(y_pred_airway) +_smooth)

    def compute_vec_np(self, y_true_airway, y_pred_airway):
        return np.sum((1.0 - y_true_airway) * y_pred_airway) / (np.sum(y_pred_airway) +_smooth)


# Combination of Two Metrics
class CombineLossTwoMetrics(Metrics):
    weights_metrics = [1.0, 3.0]

    def __init__(self, metrics1, metrics2, isMasksExclude=False):
        super(CombineLossTwoMetrics, self).__init__(isMasksExclude)
        self.metrics1 = metrics1
        self.metrics2 = metrics2
        self.name_out = '_'.join(['comb', metrics1.name_out, metrics2.name_out])

    def loss(self, y_true, y_pred):
        return self.weights_metrics[0] * self.metrics1.loss(y_true, y_pred) + \
               self.weights_metrics[1] * self.metrics2.loss(y_true, y_pred)


    # All Available Metrics
def DICTAVAILMETRICS(option):
    opts_split = option.split('_')
    if (len(opts_split) == 1):
        option = opts_split[0]
        isMasksExclude = False
    elif (len(opts_split) == 2 and opts_split[-1] == 'Masked'):
        option = opts_split[0]
        isMasksExclude = True
    else:
        return 0  # Failure
    if   (option == 'BinaryCrossEntropy'):
        return BinaryCrossEntropy(isMasksExclude=isMasksExclude)
    if   (option == 'BinaryCrossEntropyFocalLoss'):
        return BinaryCrossEntropyFocalLoss(isMasksExclude=isMasksExclude)
    elif (option == 'WeightedBinaryCrossEntropyFixedWeights'):
        return WeightedBinaryCrossEntropyFixedWeights(isMasksExclude=isMasksExclude)
    elif (option == 'WeightedBinaryCrossEntropy'):
        return WeightedBinaryCrossEntropy(isMasksExclude=isMasksExclude)
    elif (option == 'WeightedBinaryCrossEntropyFocalLoss'):
        return WeightedBinaryCrossEntropyFocalLoss(isMasksExclude=isMasksExclude)
    elif (option == 'DiceCoefficient'):
        return DiceCoefficient(isMasksExclude=isMasksExclude)
    elif (option == 'TruePositiveRate'):
        return TruePositiveRate(isMasksExclude=isMasksExclude)
    elif (option == 'TrueNegativeRate'):
        return TrueNegativeRate(isMasksExclude=isMasksExclude)
    elif (option == 'FalsePositiveRate'):
        return FalsePositiveRate(isMasksExclude=isMasksExclude)
    elif (option == 'FalseNegativeRate'):
        return FalseNegativeRate(isMasksExclude=isMasksExclude)
    else:
        return 0


def DICTAVAILLOSSFUNS(option):
    opts_split = option.split('_')
    if (opts_split[0] == 'Combine'):
        if (len(opts_split) == 3):
            option_sub1 = opts_split[1]
            option_sub2 = opts_split[2]
            isMasksExclude = False
        elif (len(opts_split) == 4 and opts_split[-1] == 'Masked'):
            option_sub1 = opts_split[1] +'_Masked'
            option_sub2 = opts_split[2] +'_Masked'
            isMasksExclude = True
        else:
            return 0  # Failure
        metrics_sub1 = DICTAVAILMETRICS(option_sub1)
        metrics_sub2 = DICTAVAILMETRICS(option_sub2)
        metrics = CombineLossTwoMetrics(metrics_sub1, metrics_sub2, isMasksExclude=isMasksExclude)
    else:
        metrics = DICTAVAILMETRICS(option)
    return metrics.loss


def DICTAVAILMETRICFUNS(option, use_in_Keras=True, set_fun_name=False):
    metrics = DICTAVAILMETRICS(option)
    if use_in_Keras:
        if set_fun_name:
            return metrics.get_renamed_compute()
        else:
            return metrics.compute
    else:
        return metrics.compute_np_safememory
