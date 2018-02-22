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


class LossFunction(object):
    def __init__(self, Metrics):
        self.Metrics = Metrics

    def compute(self, y_true, y_pred):
        return self.Metrics.compute(y_true, y_pred)


# DIFFERENT METRICS:

class Metrics(object):
    @classmethod
    def compute(cls, y_true, y_pred):
        pass

# Binary Cross entropy
class Binary_CrossEntropy(Metrics):
    @classmethod
    def compute(cls, y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    @classmethod
    def compute_home(cls, y_true, y_pred):
        return np.mean(y_true*np.log(y_pred) + (1.0-y_true)*np.log(1.0-y_pred))

# Dice coefficient
class DiceCoefficient(Metrics):
    smooth = 1
    @classmethod
    def compute(cls, y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2.0*intersection + cls.smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + cls.smooth)