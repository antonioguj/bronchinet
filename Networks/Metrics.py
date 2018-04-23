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


class LossFunction(object):
    def __init__(self, Metrics):
        self.Metrics = Metrics

    def compute(self, y_true, y_pred):
        return self.Metrics.compute_loss(y_true, y_pred)


# DIFFERENT METRICS:
class Metrics(object):

    max_size_memory_safe = 5e+08
    val_exclude = -1

    def __init__(self, isMasksExclude=False):
        self.isMasksExclude = isMasksExclude

    def compute_loss(self, y_true, y_pred):
        pass

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


# Binary Cross entropy
class BinaryCrossEntropy(Metrics):
    
    def __init__(self, isMasksExclude=False):
        super(BinaryCrossEntropy, self).__init__(isMasksExclude)

    def compute_loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def compute_vec(self, y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

    def compute_vec_np(self, y_true, y_pred):
        return np.mean(-y_true*np.log(y_pred) - (1.0-y_true)*np.log(1.0-y_pred))


# Categorical Cross entropy
class CategoricalCrossEntropy(Metrics):
    def __init__(self, isMasksExclude=False):
        super(CategoricalCrossEntropy, self).__init__(isMasksExclude)

    def compute_loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def compute_vec(self, y_true, y_pred):
        return K.mean(K.categorical_crossentropy(y_true, y_pred), axis=-1)

    def compute_vec_np(self, y_true, y_pred):
        return np.mean(-y_true * np.log(y_pred) - (1.0 - y_true) * np.log(1.0 - y_pred))


# Weighted Binary Cross entropy
class WeightedBinaryCrossEntropy(Metrics):
    weights_nomasks = [1.0, 300.0]
    weights_masks   = [1.0, 80.0]

    def __init__(self, isMasksExclude=False):
        if isMasksExclude:
            self.weights = self.weights_masks
        else:
            self.weights = self.weights_nomasks
        super(WeightedBinaryCrossEntropy, self).__init__(isMasksExclude)

    def compute_loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def compute_vec(self, y_true, y_pred):
        return K.mean(-self.weights[1]*y_true*K.log(y_pred+_eps) - self.weights[0]*(1.0-y_true)*K.log(1.0-y_pred+_eps))

    def compute_vec_np(self, y_true, y_pred):
        return np.mean(-self.weights[1]*y_true*np.log(y_pred+_eps) - self.weights[0]*(1.0-y_true)*np.log(1.0-y_pred+_eps))


# Dice coefficient
class DiceCoefficient(Metrics):
    smooth = 1

    def __init__(self, isMasksExclude=False):
        super(DiceCoefficient, self).__init__(isMasksExclude)

    def compute_loss(self, y_true, y_pred):
        return -self.compute(y_true, y_pred)

    def compute_vec(self, y_true, y_pred):
        return (2.0*K.sum(y_true * y_pred) + self.smooth) / (K.sum(y_true) + K.sum(y_pred) + self.smooth)

    def compute_vec_np(self, y_true, y_pred):
        return (2.0*np.sum(y_true * y_pred) + self.smooth) / (np.sum(y_true) + np.sum(y_pred) + self.smooth)


# True positive rate
class TruePositiveRate(Metrics):
    smooth = 1

    def __init__(self, isMasksExclude=False):
        super(TruePositiveRate, self).__init__(isMasksExclude)

    def compute_vec(self, y_true, y_pred):
        return K.sum(y_true * y_pred) / (K.sum(y_true) + self.smooth)

    def compute_vec_np(self, y_true, y_pred):
        return np.sum(y_true * y_pred) / (np.sum(y_true) + self.smooth)

# True negative rate
class TrueNegativeRate(Metrics):
    smooth = 1

    def __init__(self, isMasksExclude=False):
        super(TrueNegativeRate, self).__init__(isMasksExclude)

    def compute_vec(self, y_true, y_pred):
        return K.sum((1.0 - y_true) * (1.0 - y_pred)) / (K.sum((1.0 - y_true)) + self.smooth)

    def compute_vec_np(self, y_true, y_pred):
        return np.sum((1.0 - y_true) * (1.0 - y_pred)) / (np.sum((1.0 - y_true)) + self.smooth)

# False positive rate
class FalsePositiveRate(Metrics):
    smooth = 1

    def __init__(self, isMasksExclude=False):
        super(FalsePositiveRate, self).__init__(isMasksExclude)

    def compute_vec(self, y_true, y_pred):
        return K.sum((1.0 - y_true) * y_pred) / (K.sum((1.0 - y_true)) + self.smooth)

    def compute_vec_np(self, y_true, y_pred):
        return np.sum((1.0 - y_true) * y_pred) / (np.sum((1.0 - y_true)) + self.smooth)

# False negative rate
class FalseNegativeRate(Metrics):
    smooth = 1

    def __init__(self, isMasksExclude=False):
        super(FalseNegativeRate, self).__init__(isMasksExclude)

    def compute_vec(self, y_true, y_pred):
        return K.sum(y_true * (1.0 - y_pred)) / (K.sum(y_true) + self.smooth)

    def compute_vec_np(self, y_true, y_pred):
        return np.sum(y_true * (1.0 - y_pred)) / (np.sum(y_true) + self.smooth)



# All Available Loss Functions and Metrics
def DICTAVAILLOSSFUNS(option):
    if   (option == "BinaryCrossEntropy"):
        return BinaryCrossEntropy()
    elif (option == "BinaryCrossEntropy_Masked"):
        return BinaryCrossEntropy(isMasksExclude=True)
    elif (option == "WeightedBinaryCrossEntropy"):
        return WeightedBinaryCrossEntropy()
    elif (option == "WeightedBinaryCrossEntropy_Masked"):
        return WeightedBinaryCrossEntropy(isMasksExclude=True)
    elif (option == "CategoricalCrossEntropy"):
        return CategoricalCrossEntropy()
    else:
        return 0

def DICTAVAILMETRICS(option):
    if   (option == "DiceCoefficient"):
        return DiceCoefficient()
    elif (option == "DiceCoefficient_Masked"):
        return DiceCoefficient(isMasksExclude=True)