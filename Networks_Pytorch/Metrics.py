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
import torch
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

    def __init__(self, is_masks_exclude=False):
        self.is_masks_exclude = is_masks_exclude
        self.name_fun_out = None

    def compute(self, y_true, y_pred):
        if self.is_masks_exclude:
            return self.compute_vec_masked(y_true.view(y_true.size(0), -1),
                                           y_pred.view(y_pred.size(0), -1))
        else:
            return self.compute_vec(y_true.view(y_true.size(0), -1),
                                    y_pred.view(y_pred.size(0), -1))

    def compute_vec(self, y_true, y_pred):
        pass

    def compute_vec_masked(self, y_true, y_pred):
        pass

    def compute_np(self, y_true, y_pred):
        if self.is_masks_exclude:
            return self.compute_vec_masked_np(y_true.view(y_true.size(0), -1),
                                              y_pred.view(y_pred.size(0), -1))
        else:
            return self.compute_vec_np(y_true.view(y_true.size(0), -1),
                                       y_pred.view(y_pred.size(0), -1))

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
        return torch.where(torch.equal(y_true, self.val_exclude), torch.zeros_like(y_true), torch.ones_like(y_true))

    def get_masked_array(self, y_true, y_array):
        return torch.where(torch.equal(y_true, self.val_exclude), torch.zeros_like(y_array), y_array)

    def get_mask_np(self, y_true):
        return np.where(y_true == self.val_exclude, 0, 1)

    def get_masked_array_np(self, y_true, y_array):
        return np.where(y_true == self.val_exclude, 0, y_array)

    def loss(self, y_true, y_pred):
        pass

    def get_renamed_compute(self):
        if self.name_fun_out:
            return getattr(self, self.name_fun_out)


# mean squared error
class MeanSquared(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(MeanSquared, self).__init__(is_masks_exclude)
        self.name_fun_out  = 'mean_squared'

    def compute_vec(self, y_true, y_pred):
        return torch.mean(torch.square(y_pred - y_true))

    def compute_vec_masked(self, y_true, y_pred):
        mask = self.get_mask(y_true)
        return torch.mean(torch.square(y_pred - y_true) * mask)

    def compute_vec_np(self, y_true, y_pred):
        pass

    def compute_vec_masked_np(self, y_true, y_pred):
        pass

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def mean_squared(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# binary cross entropy
class BinaryCrossEntropy(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(BinaryCrossEntropy, self).__init__(is_masks_exclude)
        self.name_fun_out = 'bin_cross'

    def compute_vec(self, y_true, y_pred):
        return torch.mean(- y_true * torch.log(y_pred +_eps) -
                          (1.0 - y_true) * torch.log(1.0 - y_pred +_eps))

    def compute_vec_masked(self, y_true, y_pred):
        mask = self.get_mask(y_true)
        return torch.mean((- y_true * torch.log(y_pred + _eps) -
                           (1.0 - y_true) * torch.log(1.0 - y_pred + _eps)) * mask)

    def compute_vec_np(self, y_true, y_pred):
        pass

    def compute_vec_masked_np(self, y_true, y_pred):
        pass

    def loss(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def bin_cross(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# Dice coefficient
class DiceCoefficient(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(DiceCoefficient, self).__init__(is_masks_exclude)
        self.name_fun_out = 'dice'

    def compute_vec(self, y_true, y_pred):
        return (2.0*torch.sum(y_true * y_pred)) / (torch.sum(y_true) + torch.sum(y_pred) +_smooth)

    def compute_vec_masked(self, y_true, y_pred):
        return self.compute_vec(self.get_masked_array(y_true, y_true),
                                self.get_masked_array(y_true, y_pred))

    def compute_vec_np(self, y_true, y_pred):
        pass

    def compute_vec_masked_np(self, y_true, y_pred):
        pass

    def loss(self, y_true, y_pred):
        return 1.0 - self.compute(y_true, y_pred)

    def dice(self, y_true, y_pred):
        return self.compute(y_true, y_pred)