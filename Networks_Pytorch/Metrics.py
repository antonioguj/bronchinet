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
from scipy.spatial import distance
from Common.ErrorMessages import *
import numpy as np

_eps = 1e-7
_smooth = 1.0


# VARIOUS METRICS:
class Metrics(nn.Module):
    max_size_memory_safe = 5e+08
    val_exclude = -1
    count = 0
    _isUse_reference_clines = False
    _isUse_predicted_clines = False

    def __init__(self, is_masks_exclude= False):
        self.is_masks_exclude = is_masks_exclude
        self.name_fun_out = None
        super(Metrics, self).__init__()

    def forward(self, y_true, y_pred):
        return NotImplemented

    def compute_fun(self, y_true, y_pred):
        return NotImplemented

    def compute_fun_np(self, y_true, y_pred):
        return NotImplemented

    def compute_masked(self, y_true, y_pred):
        return self.compute_fun(self.get_masked_array(y_true, y_true),
                                self.get_masked_array(y_true, y_pred))

    def compute_masked_np(self, y_true, y_pred):
        return self.compute_fun_np(self.get_masked_array_np(y_true, y_true),
                                   self.get_masked_array_np(y_true, y_pred))

    def compute(self, y_true, y_pred):
        if self.is_masks_exclude:
            return self.compute_masked(y_true, y_pred)
        else:
            return self.compute_fun(y_true, y_pred)

    def compute_np(self, y_true, y_pred):
        if self.is_masks_exclude:
            return self.compute_masked_np(y_true, y_pred)
        else:
            return self.compute_fun_np(y_true, y_pred)

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
        return torch.where(y_true == self.val_exclude, torch.zeros_like(y_true), torch.ones_like(y_true))

    def get_masked_array(self, y_true, y_array):
        return torch.where(y_true == self.val_exclude, torch.zeros_like(y_array), y_array)

    def get_mask_np(self, y_true):
        return np.where(y_true == self.val_exclude, 0, 1)

    def get_masked_array_np(self, y_true, y_array):
        return np.where(y_true == self.val_exclude, 0, y_array)

    def loss(self, y_true, y_pred):
        return self.forward(y_true, y_pred)


# mean squared error
class MeanSquared(Metrics):

    def __init__(self, is_masks_exclude= False):
        super(MeanSquared, self).__init__(is_masks_exclude)
        self.name_fun_out  = 'mean_squared'

    def compute_fun(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        return torch.mean(torch.square(y_pred - y_true))

    def compute_masked(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        mask = self.get_mask(y_true)
        return torch.mean(torch.square(y_pred - y_true) * mask)

    def forward(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def mean_squared(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# binary cross entropy
class BinaryCrossEntropy(Metrics):

    def __init__(self, is_masks_exclude= False):
        super(BinaryCrossEntropy, self).__init__(is_masks_exclude)
        self.name_fun_out = 'bin_cross'

    def compute_fun(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        return torch.mean(- y_true * torch.log(y_pred +_eps) -
                          (1.0 - y_true) * torch.log(1.0 - y_pred +_eps))

    def compute_masked(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        mask = self.get_mask(y_true)
        return torch.mean((- y_true * torch.log(y_pred +_eps) -
                           (1.0 - y_true) * torch.log(1.0 - y_pred +_eps)) * mask)

    def forward(self, y_true, y_pred):
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

    def compute_fun(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        return torch.mean(- self.weights[1] * y_true * torch.log(y_pred +_eps) -
                          self.weights[0] * (1.0 - y_true) * torch.log(1.0 - y_pred +_eps))

    def compute_masked(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        mask = self.get_mask(y_true)
        return torch.mean((- self.weights[1] * y_true * torch.log(y_pred +_eps) -
                           self.weights[0] * (1.0 - y_true) * torch.log(1.0 - y_pred +_eps)) * mask)

    def forward(self, y_true, y_pred):
        return self.compute(y_true, y_pred)

    def wei_bin_cross_fixed(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# Dice coefficient
class DiceCoefficient(Metrics):

    def __init__(self, is_masks_exclude= False):
        super(DiceCoefficient, self).__init__(is_masks_exclude)
        self.name_fun_out = 'dice'

    def compute_fun(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        return (2.0 * torch.sum(y_true * y_pred)) / (torch.sum(y_true) + torch.sum(y_pred) +_smooth)

    def compute_fun_np(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        return (2.0*np.sum(y_true * y_pred)) / (np.sum(y_true) + np.sum(y_pred) +_smooth)

    def forward(self, y_true, y_pred):
        return 1.0 - self.compute(y_true, y_pred)

    def dice(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


# true positive rate
class TruePositiveRate(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(TruePositiveRate, self).__init__(is_masks_exclude)
        self.name_fun_out = 'tpr'

    def compute_fun(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        return torch.sum(y_true * y_pred) / (torch.sum(y_true) +_smooth)

    def compute_fun_np(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
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

    def compute_fun(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        return torch.sum((1.0 - y_true) * (1.0 - y_pred)) / (torch.sum((1.0 - y_true)) +_smooth)

    def compute_fun_np(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
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

    def compute_fun(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        return torch.sum((1.0 - y_true) * y_pred) / (torch.sum((1.0 - y_true)) +_smooth)

    def compute_fun_np(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
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

    def compute_fun(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        return torch.sum(y_true * (1.0 - y_pred)) / (torch.sum(y_true) +_smooth)

    def compute_fun_np(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
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

    def compute_fun(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        return torch.sum(y_true * y_pred) / (torch.sum(y_true) +_smooth)

    def compute_fun_np(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        return np.sum(y_true * y_pred) / (np.sum(y_true) +_smooth)


# airways volume leakage (percentage of voxels from predicted airways found outside the ground-truth airways)
class AirwayVolumeLeakage(Metrics):

    def __init__(self, is_masks_exclude=False):
        super(AirwayVolumeLeakage, self).__init__(is_masks_exclude)
        self.name_fun_out = 'volume_leakage'

    def compute_fun(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        return torch.sum((1.0 - y_true) * y_pred) / (torch.sum(y_pred) +_smooth)

    def compute_fun_np(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        return np.sum((1.0 - y_true) * y_pred) / (np.sum(y_pred) +_smooth)


class AirwayCompletenessModified(Metrics):
    _isUse_reference_clines = False
    _isUse_predicted_clines = True

    def __init__(self, is_masks_exclude=False):
        super(AirwayCompletenessModified, self).__init__(is_masks_exclude)
        self.name_fun_out = 'completeness_mod'

    def compute_fun(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        return torch.sum(y_true * y_pred) / (torch.sum(y_pred) +_smooth)

    def compute_fun_np_correct(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        return np.sum(y_true * y_pred) / (np.sum(y_pred) +_smooth)

    def compute_fun_np_correct(self, y_true, y_pred, y_true_cl):
        if self.is_masks_exclude:
            y_true = self.get_masked_array_np(y_true, y_true).flatten()
            y_pred = self.get_masked_array_np(y_true, y_pred).flatten()
            y_true_cl = self.get_masked_array_np(y_true, y_true_cl).flatten()
            return np.sum(y_true * y_pred) / (np.sum(y_true_cl) + _smooth)
        else:
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            y_true_cl = y_true_cl.flatten()
            return np.sum(y_true * y_pred) / (np.sum(y_true_cl) +_smooth)


class AirwayCentrelineLeakage(Metrics):
    _isUse_reference_clines = False
    _isUse_predicted_clines = True

    def __init__(self, is_masks_exclude=False):
        super(AirwayCentrelineLeakage, self).__init__(is_masks_exclude)
        self.name_fun_out = 'centreline_leakage'

    def compute_fun(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        return torch.sum((1.0 - y_true) * y_pred) / (torch.sum(y_pred) +_smooth)

    def compute_fun_np(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        return np.sum((1.0 - y_true) * y_pred) / (np.sum(y_pred) +_smooth)


# airways centreline False Positive distance error
class AirwayCentrelineFalsePositiveDistanceError(Metrics):
    _isUse_reference_clines = True
    _isUse_predicted_clines = True

    def __init__(self, is_masks_exclude=False):
        super(AirwayCentrelineFalsePositiveDistanceError, self).__init__(is_masks_exclude)
        self.name_fun_out = 'cenline_DFP_error'

    @staticmethod
    def get_voxel_scaling(y_array):
        #return np.diag(y_array.affine)[:3]
        return np.asarray([1.0, 1.0, 1.0])

    @classmethod
    def get_centreline_coords(cls, y_array):
        return np.asarray(np.argwhere(y_array > 0)) * cls.get_voxel_scaling(y_array)

    # def compute_fun(self, y_true, y_pred):
    #     y_true = self.get_centreline_coords(y_true)
    #     y_pred = self.get_centreline_coords(y_pred)
    #     dist_y = distance.cdist(y_pred, y_true)
    #     return torch.mean(torch.min(dist_y, axis=1))

    def compute_fun_np(self, y_true, y_pred):
        y_true = self.get_centreline_coords(y_true)
        y_pred = self.get_centreline_coords(y_pred)
        dist_y = distance.cdist(y_pred, y_true)
        return np.mean(np.min(dist_y, axis=1))


# airways centreline False Negative distance error
class AirwayCentrelineFalseNegativeDistanceError(Metrics):
    _isUse_reference_clines = True
    _isUse_predicted_clines = True

    def __init__(self, is_masks_exclude=False):
        super(AirwayCentrelineFalseNegativeDistanceError, self).__init__(is_masks_exclude)
        self.name_fun_out = 'cenline_DFN_error'

    @staticmethod
    def get_voxel_scaling(y_array):
        #return np.diag(y_array.affine)[:3]
        return np.asarray([1.0, 1.0, 1.0])

    @classmethod
    def get_centreline_coords(cls, y_array):
        return np.asarray(np.argwhere(y_array > 0)) * cls.get_voxel_scaling(y_array)

    # def compute_fun(self, y_true, y_pred):
    #     y_true = self.get_centreline_coords(y_true)
    #     y_pred = self.get_centreline_coords(y_pred)
    #     dist_y = distance.cdist(y_pred, y_true)
    #     return torch.mean(torch.min(dist_y, axis=0))

    def compute_fun_np(self, y_true, y_pred):
        y_true = self.get_centreline_coords(y_true)
        y_pred = self.get_centreline_coords(y_pred)
        dist_y = distance.cdist(y_pred, y_true)
        return np.mean(np.min(dist_y, axis=0))



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
def DICTAVAILMETRICLASS(option,
                        is_masks_exclude= False):
    list_metric_avail = ['MeanSquared',
                         'BinaryCrossEntropy', 'WeightedBinaryCrossEntropy',
                         'DiceCoefficient',
                         'TruePositiveRate', 'TrueNegativeRate', 'FalsePositiveRate', 'FalseNegativeRate',
                         'AirwayCompleteness', 'AirwayVolumeLeakage',
                         'AirwayCompletenessModified', 'AirwayCentrelineLeakage',
                         'AirwayCentrelineFalsePositiveDistanceError', 'AirwayCentrelineFalseNegativeDistanceError']

    if   (option == 'MeanSquared'):
        return MeanSquared(is_masks_exclude= is_masks_exclude)
    elif (option == 'BinaryCrossEntropy'):
        return BinaryCrossEntropy(is_masks_exclude= is_masks_exclude)
    elif (option == 'WeightedBinaryCrossEntropy'):
        return WeightedBinaryCrossEntropyFixedWeights(is_masks_exclude=is_masks_exclude)
    elif (option == 'WeightedBinaryCrossEntropyFixedWeights'):
        return WeightedBinaryCrossEntropyFixedWeights(is_masks_exclude=is_masks_exclude)
    elif (option == 'DiceCoefficient'):
        return DiceCoefficient(is_masks_exclude= is_masks_exclude)
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
    elif (option == 'AirwayCompletenessModified'):
        return AirwayCompletenessModified(is_masks_exclude=is_masks_exclude)
    elif (option == 'AirwayCentrelineLeakage'):
        return AirwayCentrelineLeakage(is_masks_exclude=is_masks_exclude)
    elif (option == 'AirwayCentrelineFalsePositiveDistanceError'):
        return AirwayCentrelineFalsePositiveDistanceError(is_masks_exclude=is_masks_exclude)
    elif (option == 'AirwayCentrelineFalseNegativeDistanceError'):
        return AirwayCentrelineFalseNegativeDistanceError(is_masks_exclude=is_masks_exclude)
    else:
        message = 'Metric \'%s\' chosen not found. Metrics available: \'%s\'...' %(option, ', '.join(list_metric_avail))
        CatchErrorException(message)
        return NotImplemented


def DICTAVAILLOSSFUNS(option, is_masks_exclude= False, option2_combine= None):
    if option2_combine:
        metrics_sub1 = DICTAVAILMETRICLASS(option, is_masks_exclude)
        metrics_sub2 = DICTAVAILMETRICLASS(option2_combine, is_masks_exclude)
        return CombineLossTwoMetrics(metrics_sub1, metrics_sub2, is_masks_exclude= is_masks_exclude)
    else:
        return DICTAVAILMETRICLASS(option, is_masks_exclude)


def DICTAVAILMETRICFUNS(option, is_masks_exclude=False):
    return DICTAVAILMETRICLASS(option, is_masks_exclude)
