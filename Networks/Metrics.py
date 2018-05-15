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

_eps = 1.0e-06
_smooth = 1


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
            return self.compute_vec(K.flatten(self.get_masked_array(y_true, y_true)),
                                    K.flatten(self.get_masked_array(y_true, y_pred)))
        else:
            return self.compute_vec(K.flatten(y_true), K.flatten(y_pred))

    def compute_vec(self, y_true, y_pred):
        pass

    def compute_np(self, y_true, y_pred):
        if self.isMasksExclude:
            return self.compute_vec_np(self.get_masked_array_np(y_true, y_true).flatten(),
                                       self.get_masked_array_np(y_true, y_pred).flatten())
        else:
            return self.compute_vec_np(y_true.flatten(), y_pred.flatten())

    def compute_vec_np(self, y_true, y_pred):
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
        return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

    def compute_vec_np(self, y_true, y_pred):
        return np.mean(-y_true * np.log(y_pred +_eps) - (1.0 - y_true) * np.log(1.0 - y_pred +_eps))

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def bin_cross(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# Weighted Binary Cross entropy
class WeightedBinaryCrossEntropy(Metrics):
    #weights_nomasks = [1.0, 80.0]
    #weights_masks   = [1.0, 300.0]

    def __init__(self, isMasksExclude=False):
        #if isMasksExclude:
        #    self.weights = self.weights_masks
        #else:
        #    self.weights = self.weights_nomasks
        super(WeightedBinaryCrossEntropy, self).__init__(isMasksExclude)
        self.name_out = 'wei_bin_cross'

    def get_weights(self, y_true):
        num_class_1 = K.tf.count_nonzero(y_true, dtype=K.tf.int32)
        num_class_0 = K.tf.size(y_true) - num_class_1
        return (1.0, K.cast(num_class_0, dtype=K.tf.float32) / (K.cast(num_class_1, dtype=K.tf.float32) + K.variable(_eps)))

    def get_weights_np(self, y_true):
        num_class_1 = np.count_nonzero(y_true == 1)
        num_class_0 = np.size(y_true) - num_class_1
        return (1.0, num_class_0 / (float(num_class_1) +_eps))

    def compute_vec(self, y_true, y_pred):
        weights = self.get_weights(y_true)
        return K.mean(-weights[1] * y_true * K.log(y_pred +_eps) - weights[0] * (1.0 - y_true) * K.log(1.0 - y_pred +_eps))

    def compute_vec_np(self, y_true, y_pred):
        weights = self.get_weights_np(y_true)
        return np.mean(-weights[1] * y_true * np.log(y_pred +_eps) - weights[0] * (1.0 - y_true) * np.log(1.0 - y_pred +_eps))

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def wei_bin_cross(self, y_true, y_pred):
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

    def compute_vec_np(self, y_true, y_pred):
        return (2.0*np.sum(y_true * y_pred)) / (np.sum(y_true) + np.sum(y_pred) +_smooth)

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

    def compute_vec_np(self, y_true, y_pred):
        return np.sum(y_true * y_pred) / (np.sum(y_true) +_smooth)

    def tpr(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# True negative rate
class TrueNegativeRate(Metrics):

    def __init__(self, isMasksExclude=False):
        super(TrueNegativeRate, self).__init__(isMasksExclude)
        self.name_out = 'tnr'

    def compute_vec(self, y_true, y_pred):
        return K.sum((1.0 - y_true) * (1.0 - y_pred)) / (K.sum((1.0 - y_true)) +_smooth)

    def compute_vec_np(self, y_true, y_pred):
        return np.sum((1.0 - y_true) * (1.0 - y_pred)) / (np.sum((1.0 - y_true)) +_smooth)

    def tnr(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# False positive rate
class FalsePositiveRate(Metrics):

    def __init__(self, isMasksExclude=False):
        super(FalsePositiveRate, self).__init__(isMasksExclude)
        self.name_out = 'fpr'

    def compute_vec(self, y_true, y_pred):
        return K.sum((1.0 - y_true) * y_pred) / (K.sum((1.0 - y_true)) +_smooth)

    def compute_vec_np(self, y_true, y_pred):
        return np.sum((1.0 - y_true) * y_pred) / (np.sum((1.0 - y_true)) +_smooth)

    def fpr(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# False negative rate
class FalseNegativeRate(Metrics):

    def __init__(self, isMasksExclude=False):
        super(FalseNegativeRate, self).__init__(isMasksExclude)
        self.name_out = 'fnr'

    def compute_vec(self, y_true, y_pred):
        return K.sum(y_true * (1.0 - y_pred)) / (K.sum(y_true) +_smooth)

    def compute_vec_np(self, y_true, y_pred):
        return np.sum(y_true * (1.0 - y_pred)) / (np.sum(y_true) +_smooth)

    def fnr(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# All Available Loss Functions and Metrics
def DICTAVAILLOSSFUNS(option):
    if   (option == 'BinaryCrossEntropy'):
        return BinaryCrossEntropy().loss
    elif (option == 'BinaryCrossEntropy_Masked'):
        return BinaryCrossEntropy(isMasksExclude=True).loss
    elif (option == 'WeightedBinaryCrossEntropy'):
        return WeightedBinaryCrossEntropy().loss
    elif (option == 'WeightedBinaryCrossEntropy_Masked'):
        return WeightedBinaryCrossEntropy(isMasksExclude=True).loss
    elif (option == 'CategoricalCrossEntropy'):
        return 'categorical_crossentropy'
    elif (option == 'DiceCoefficient'):
        return DiceCoefficient().loss
    elif (option == 'DiceCoefficient_Masked'):
        return DiceCoefficient(isMasksExclude=True).loss
    else:
        return 0


def DICTAVAILMETRICS(option, use_in_Keras=True, set_fun_name=False):
    if   (option == 'BinaryCrossEntropy'):
        metrics = BinaryCrossEntropy()
    elif (option == 'BinaryCrossEntropy_Masked'):
        metrics = BinaryCrossEntropy(isMasksExclude=True)
    elif (option == 'WeightedBinaryCrossEntropy'):
        metrics = WeightedBinaryCrossEntropy()
    elif (option == 'WeightedBinaryCrossEntropy_Masked'):
        metrics = WeightedBinaryCrossEntropy(isMasksExclude=True)
    elif (option == 'DiceCoefficient'):
        metrics = DiceCoefficient()
    elif (option == 'DiceCoefficient_Masked'):
        metrics = DiceCoefficient(isMasksExclude=True)
    elif (option == 'TruePositiveRate'):
        metrics = TruePositiveRate()
    elif (option == 'TruePositiveRate_Masked'):
        metrics = TruePositiveRate(isMasksExclude=True)
    elif (option == 'TrueNegativeRate'):
        metrics = TrueNegativeRate()
    elif (option == 'TrueNegativeRate_Masked'):
        metrics = TrueNegativeRate(isMasksExclude=True)
    elif (option == 'FalsePositiveRate'):
        metrics = FalsePositiveRate()
    elif (option == 'FalsePositiveRate_Masked'):
        metrics = FalsePositiveRate(isMasksExclude=True)
    elif (option == 'FalseNegativeRate'):
        metrics = FalseNegativeRate()
    elif (option == 'FalseNegativeRate_Masked'):
        metrics = FalseNegativeRate(isMasksExclude=True)
    else:
        return 0
    if use_in_Keras:
        if set_fun_name:
            return metrics.get_renamed_compute()
        else:
            return metrics.compute
    else:
        return metrics.compute_np_safememory