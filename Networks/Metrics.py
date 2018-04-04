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
    @classmethod
    def compute(cls, y_true, y_pred):
        pass
    @classmethod
    def compute_loss(cls, y_true, y_pred):
        pass


# Binary Cross entropy
class BinaryCrossEntropy(Metrics):
    @classmethod
    def compute(cls, y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

    @classmethod
    def compute_loss(cls, y_true, y_pred):
        return cls.compute(y_true, y_pred)

    @classmethod
    def compute_home(cls, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        return np.mean(-y_true*np.log(y_pred) - (1.0-y_true)*np.log(1.0-y_pred))


# Weighted Binary Cross entropy
class WeightedBinaryCrossEntropy(Metrics):
    weights = [1.0, 300.0]

    @classmethod
    def compute(cls, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        return K.mean(-cls.weights[1]*y_true*K.log(y_pred+_eps) - cls.weights[0]*(1.0-y_true)*K.log(1.0-y_pred+_eps))

    @classmethod
    def compute_loss(cls, y_true, y_pred):
        return cls.compute(y_true, y_pred)

    @classmethod
    def compute_home(cls, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        return np.mean(-cls.weights[1]*y_true*np.log(y_pred+_eps) - cls.weights[0]*(1.0-y_true)*np.log(1.0-y_pred+_eps))


# Weighted Binary Cross entropy. Exclude voxels outside the mask
class WeightedBinaryCrossEntropy_Masks(Metrics):
    weights = [1.0, 100.0]
    val_exclude = -1

    @classmethod
    def get_mask(cls, y_true):
        return K.not_equal(y_true, cls.val_exclude)

    @classmethod
    def get_mask_home(cls, y_true):
        return np.where(y_true != cls.val_exclude, 1, 0)

    @classmethod
    def compute(cls, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        mask   = cls.get_mask(y_true)
        return K.mean((-cls.weights[1]*y_true*K.log(y_pred+_eps) - cls.weights[0]*(1.0-y_true)*K.log(1.0-y_pred+_eps))*mask)

    @classmethod
    def compute_loss(cls, y_true, y_pred):
        return cls.compute(y_true, y_pred)

    @classmethod
    def compute_home(cls, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        mask   = cls.get_mask_home(y_true)
        return np.mean((-cls.weights[1]*y_true*np.log(y_pred+_eps) - cls.weights[0]*(1.0-y_true)*np.log(1.0-y_pred+_eps))*mask)


# Dice coefficient
class DiceCoefficient(Metrics):
    smooth = 1

    @classmethod
    def compute(cls, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        intersection = K.sum(y_true * y_pred)
        return (2.0*intersection + cls.smooth) / (K.sum(y_true) + K.sum(y_pred) + cls.smooth)

    @classmethod
    def compute_loss(cls, y_true, y_pred):
        return -cls.compute(y_true, y_pred)

    @classmethod
    def compute_home(cls, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        intersection = np.sum(y_true * y_pred)
        return (2.0*intersection + cls.smooth) / (np.sum(y_true) + np.sum(y_pred) + cls.smooth)


# Dice coefficient. Exclude voxels outside the mask
class DiceCoefficient_Masks(Metrics):
    smooth = 1
    val_exclude = -1

    @classmethod
    def get_mask(cls, y_true):
        return K.not_equal(y_true, cls.val_exclude)

    @classmethod
    def get_mask_home(cls, y_true):
        return np.where(y_true != cls.val_exclude, 1, 0)

    @classmethod
    def compute(cls, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        mask   = cls.get_mask(y_true)
        intersection = K.sum(y_true * y_pred * mask)
        return (2.0*intersection + cls.smooth) / (K.sum(y_true * mask) + K.sum(y_pred * mask) + cls.smooth)

    @classmethod
    def compute_loss(cls, y_true, y_pred):
        return -cls.compute(y_true, y_pred)

    @classmethod
    def compute_home(cls, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        mask   = cls.get_mask_home(y_true)
        intersection = np.sum(y_true * y_pred * mask)
        return (2.0*intersection + cls.smooth) / (np.sum(y_true * mask) + np.sum(y_pred * mask) + cls.smooth)