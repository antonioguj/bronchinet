#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
import numpy as np

_eps = 1e-7
_smooth = 1.0


# class LossFunction(nn.Module):
#     def __init__(self, metrics):
#         super(LossFunction, self).__init__()
#         self.metrics = metrics
#
#     def forward(self, y_true, y_pred):
#         return self.metrics.loss(y_true, y_pred)
#
#     def compute(self, y_true, y_pred):
#         return self.metrics.loss(y_true, y_pred)


# DIFFERENT METRICS:
class Metrics(nn.Module):

    max_size_memory_safe = 5e+08
    val_exclude = -1
    count = 0

    def __init__(self, is_masks_exclude= False):
        self.is_masks_exclude = is_masks_exclude
        self.name_fun_out = None

        super(Metrics, self).__init__()

    def forward(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def compute(self, y_true, y_pred):
        if self.is_masks_exclude:
            return self.compute_vec_masked(y_true, y_pred)
        else:
            return self.compute_vec(y_true, y_pred)

    def compute_vec(self, y_true, y_pred):
        raise NotImplemented

    def compute_vec_masked(self, y_true, y_pred):
        raise NotImplemented

    # def compute_np(self, y_true, y_pred):
    #     if self.is_masks_exclude:
    #         return self.compute_vec_masked_np(y_true.flatten(), y_pred.flatten())
    #     else:
    #         return self.compute_vec_np(y_true.flatten(), y_pred.flatten())
    #
    # def compute_vec_np(self, y_true, y_pred):
    #     raise NotImplemented
    #
    # def compute_vec_masked_np(self, y_true, y_pred):
    #     raise NotImplemented
    #
    # def get_mask(self, y_true):
    #     return torch.where(y_true == self.val_exclude,
    #                        torch.zeros_like(y_true),
    #                        torch.ones_like(y_true))
    #
    # def get_masked_array(self, y_true, y_array):
    #     return torch.where(y_true == self.val_exclude,
    #                        torch.zeros_like(y_array),
    #                        y_array)
    #
    # def get_mask_np(self, y_true):
    #     return np.where(y_true == self.val_exclude, 0, 1)
    #
    # def get_masked_array_np(self, y_true, y_array):
    #     return np.where(y_true == self.val_exclude, 0, y_array)

    def loss(self, y_true, y_pred):
        raise NotImplemented

    def get_renamed_compute(self):
        if self.name_fun_out:
            return getattr(self, self.name_fun_out)


# mean squared error
class MeanSquared(Metrics):

    def __init__(self, is_masks_exclude= False):
        super(MeanSquared, self).__init__(is_masks_exclude)
        self.name_fun_out  = 'mean_squared'

    def compute_vec(self, y_true, y_pred):
        return torch.mean(torch.square(y_pred - y_true))

    def compute_vec_masked(self, y_true, y_pred):
        mask = self.get_mask(y_true)
        return torch.mean(torch.square(y_pred - y_true) * mask)

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def mean_squared(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# binary cross entropy
class BinaryCrossEntropy(Metrics):

    def __init__(self, is_masks_exclude= False):
        super(BinaryCrossEntropy, self).__init__(is_masks_exclude)
        self.name_fun_out = 'bin_cross'

    def compute_vec(self, y_true, y_pred):
        return torch.mean(- y_true * torch.log(y_pred +_eps) -
                          (1.0 - y_true) * torch.log(1.0 - y_pred +_eps))

    def compute_vec_masked(self, y_true, y_pred):
        mask = self.get_mask(y_true)
        return torch.mean((- y_true * torch.log(y_pred +_eps) -
                           (1.0 - y_true) * torch.log(1.0 - y_pred +_eps)) * mask)

    def forward(self, y_true, y_pred):
        return torch.mean(- y_true * torch.log(y_pred +_eps) -
                          (1.0 - y_true) * torch.log(1.0 - y_pred +_eps))

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def bin_cross(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# weighted binary cross entropy
class WeightedBinaryCrossEntropyFixedWeights(Metrics):
    weights_noMasksExclude = [1.0, 80.0]
    weights_masksExclude = [1.0, 300.0]  # for LUVAR data
    # weights_masksExclude = [1.0, 361.0]  # for DLCST data

    def __init__(self, is_masks_exclude=False):
        if is_masks_exclude:
            self.weights = self.weights_masksExclude
        else:
            self.weights = self.weights_noMasksExclude
        super(WeightedBinaryCrossEntropyFixedWeights, self).__init__(is_masks_exclude)
        self.name_fun_out = 'wei_bin_cross_fixed'

    def compute_vec(self, y_true, y_pred):
        return torch.mean(- self.weights[1] * y_true * torch.log(y_pred +_eps) -
                          self.weights[0] * (1.0 - y_true) * torch.log(1.0 - y_pred +_eps))

    def compute_vec_masked(self, y_true, y_pred):
        mask = self.get_mask(y_true)
        return torch.mean((- self.weights[1] * y_true * torch.log(y_pred +_eps) -
                           self.weights[0] * (1.0 - y_true) * torch.log(1.0 - y_pred +_eps)) * mask)

    def forward(self, y_true, y_pred):
        return torch.mean(- self.weights[1] * y_true * torch.log(y_pred +_eps) -
                          self.weights[0] * (1.0 - y_true) * torch.log(1.0 - y_pred +_eps))

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def wei_bin_cross_fixed(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# Dice coefficient
class DiceCoefficient(Metrics):

    def __init__(self, is_masks_exclude= False):
        super(DiceCoefficient, self).__init__(is_masks_exclude)
        self.name_fun_out = 'dice'

    def compute_vec(self, y_true, y_pred):
        return (2.0 * torch.sum(y_true * y_pred)) / (torch.sum(y_true) + torch.sum(y_pred) +_smooth)

    def compute_vec_masked(self, y_true, y_pred):
        return self.compute_vec(self.get_masked_array(y_true, y_true),
                                self.get_masked_array(y_true, y_pred))

    def forward(self, y_true, y_pred):
        return 1.0 - (2.0 * torch.sum(y_true * y_pred)) / (torch.sum(y_true) + torch.sum(y_pred) + _smooth)

    def loss(self, y_true, y_pred):
        return 1.0 - self.compute(y_true, y_pred)

    def dice(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# combination of two metrics
class CombineLossTwoMetrics(Metrics):
    weights_metrics = [1.0, 3.0]

    def __init__(self, metrics1, metrics2, is_masks_exclude= False):
        super(CombineLossTwoMetrics, self).__init__(is_masks_exclude)
        self.metrics1 = metrics1
        self.metrics2 = metrics2
        self.name_fun_out = '_'.join(['comb', metrics1.name_fun_out, metrics2.name_fun_out])

    def loss(self, y_true, y_pred):
        return self.weights_metrics[0] * self.metrics1.loss(y_true, y_pred) + \
               self.weights_metrics[1] * self.metrics2.loss(y_true, y_pred)


# all available metrics
def DICTAVAILMETRICS(option, is_masks_exclude= False):
    if   (option == 'MeanSquared'):
        return MeanSquared(is_masks_exclude= is_masks_exclude)
    elif (option == 'BinaryCrossEntropy'):
        return BinaryCrossEntropy(is_masks_exclude= is_masks_exclude)
    elif (option == 'WeightedBinaryCrossEntropy'):
        return WeightedBinaryCrossEntropyFixedWeights(is_masks_exclude=is_masks_exclude)
    elif (option == 'DiceCoefficient'):
        return DiceCoefficient(is_masks_exclude= is_masks_exclude)
    else:
        raise NotImplemented


def DICTAVAILLOSSFUNS(option, is_masks_exclude= False, option2_combine= None):
    if option2_combine:
        metrics_sub1 = DICTAVAILMETRICS(option, is_masks_exclude)
        metrics_sub2 = DICTAVAILMETRICS(option2_combine, is_masks_exclude)
        return CombineLossTwoMetrics(metrics_sub1, metrics_sub2, is_masks_exclude= is_masks_exclude)
    else:
        return DICTAVAILMETRICS(option, is_masks_exclude)


def DICTAVAILMETRICFUNS(option, is_masks_exclude=False, use_in_Keras=True, set_fun_name=False):
    metrics = DICTAVAILMETRICS(option, is_masks_exclude)
    if use_in_Keras:
        if set_fun_name:
            return metrics.get_renamed_compute()
        else:
            return metrics.compute
    else:
        raise NotImplemented