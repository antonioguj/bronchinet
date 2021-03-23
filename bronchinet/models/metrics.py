
from typing import Tuple
import numpy as np
from scipy.spatial import distance
#from scipy.stats import signaltonoise
from skimage.metrics import structural_similarity as ssim

_EPS = 1.0e-7
_SMOOTH = 1.0

LIST_AVAIL_METRICS = ['MeanSquaredError',
                      'MeanSquaredErrorLogarithmic',
                      'BinaryCrossEntropy',
                      'WeightedBinaryCrossEntropy',
                      'WeightedBinaryCrossEntropyFixedWeights',
                      'DiceCoefficient',
                      'TruePositiveRate',
                      'TrueNegativeRate',
                      'FalsePositiveRate',
                      'FalseNegativeRate',
                      'AirwayCompleteness',
                      'AirwayVolumeLeakage',
                      'AirwayCentrelineLeakage',
                      'AirwayTreeLength',
                      'AirwayCentrelineDistanceFalsePositiveError',
                      'AirwayCentrelineDistanceFalseNegativeError',
                      'SNR',
                      'PSNR',
                      'SSIM',
                      ]


class MetricBase(object):
    _max_size_memory_safe = 5e+08
    _value_mask_exclude = -1
    _is_use_ytrue_cenlines = False
    _is_use_ypred_cenlines = False
    _is_use_img_voxel_size = False

    def __init__(self, is_mask_exclude: bool = False) -> None:
        self._is_mask_exclude = is_mask_exclude
        self._name_fun_out = None

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if self._is_mask_exclude:
            return self._compute_masked(y_true.flatten(), y_pred.flatten())
        else:
            return self._compute(y_true.flatten(), y_pred.flatten())

    def compute_safememory(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if (y_true.size > self._max_size_memory_safe):
            # if arrays are too large, split then in two and compute metrics twice, and return size-weighted metrics
            totaldim_0 = y_true.shape[0]
            metrics_1 = self.compute(y_true[0:totaldim_0 / 2], y_pred[:totaldim_0 / 2])
            metrics_2 = self.compute(y_true[totaldim_0 / 2:], y_pred[totaldim_0 / 2:])
            size_1 = y_true[0:totaldim_0/2].size
            size_2 = y_true[totaldim_0/2:].size
            return (metrics_1 * size_1 + metrics_2 * size_2)/(size_1 + size_2)
        else:
            return self.compute(y_true, y_pred)

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _compute_masked(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self._compute(self._get_masked_input(y_true, y_true),
                             self._get_masked_input(y_pred, y_true))

    def _get_mask(self, y_true: np.ndarray) -> np.ndarray:
        return np.where(y_true == self._value_mask_exclude, 0, 1)

    def _get_masked_input(self, y_input: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.where(y_true == self._value_mask_exclude, 0, y_input)

    def set_voxel_size(self, voxel_size: np.ndarray) -> None:
        self._voxel_size = np.array(voxel_size)


class CombineTwoMetrics(MetricBase):

    def __init__(self, metrics_1: MetricBase, metrics_2: MetricBase, weight_metric2over1: float = 1.0) -> None:
        super(CombineTwoMetrics, self).__init__(False)
        self._metrics_1 = metrics_1
        self._metrics_2 = metrics_2
        self._weight_metric2over1 = weight_metric2over1
        self._name_fun_out = '_'.join(['combi', metrics_1._name_fun_out, metrics_2._name_fun_out])

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self._metrics_1.compute(y_true, y_pred) + self._weight_metric2over1 * self._metrics_2.compute(y_true, y_pred)

    def compute_safememory(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self._metrics_1.compute_safememory(y_true, y_pred) + self._weight_metric2over1 * self._metrics_2.compute_safememory(y_true, y_pred)


class MeanSquaredError(MetricBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MeanSquaredError, self).__init__(is_mask_exclude)
        self._name_fun_out  = 'mean_squared'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.square(y_pred - y_true))

    def _compute_masked(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        mask = self._get_mask(y_true)
        return np.mean(np.square(y_pred - y_true) * mask)


class MeanSquaredErrorLogarithmic(MetricBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MeanSquaredErrorLogarithmic, self).__init__(is_mask_exclude)
        self._name_fun_out  = 'mean_squared_log'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.square(np.log(np.clip(y_pred, _EPS, None) + 1.0) -
                                 np.log(np.clip(y_true, _EPS, None) + 1.0)))

    def _compute_masked(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        mask = self._get_mask(y_true)
        return np.mean(np.square(np.log(np.clip(y_pred, _EPS, None) + 1.0) -
                                 np.log(np.clip(y_true, _EPS, None) + 1.0)) * mask)


class BinaryCrossEntropy(MetricBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(BinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_fun_out = 'bin_cross'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(- y_true * np.log(y_pred + _EPS)
                       - (1.0 - y_true) * np.log(1.0 - y_pred + _EPS))

    def _compute_masked(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        mask = self._get_mask(y_true)
        return np.mean((- y_true * np.log(y_pred + _EPS)
                        - (1.0 - y_true) * np.log(1.0 - y_pred + _EPS)) * mask)


class WeightedBinaryCrossEntropy(MetricBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(WeightedBinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_fun_out = 'weight_bin_cross'

    def _get_weights(self, y_true: np.ndarray) -> Tuple[float, float]:
        num_class_1 = np.count_nonzero(y_true == 1)
        num_class_0 = np.count_nonzero(y_true == 0)
        return (1.0, num_class_0 / (float(num_class_1) + _EPS))

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        weights = self._get_weights(y_true)
        return np.mean(- weights[1] * y_true * np.log(y_pred + _EPS)
                       - weights[0] * (1.0 - y_true) * np.log(1.0 - y_pred + _EPS))

    def _compute_masked(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        weights = self._get_weights(y_true)
        mask = self._get_mask(y_true)
        return np.mean((- weights[1] * y_true * np.log(y_pred + _EPS)
                        - weights[0] * (1.0 - y_true) * np.log(1.0 - y_pred + _EPS)) * mask)


class WeightedBinaryCrossEntropyFixedWeights(WeightedBinaryCrossEntropy):
    weights_no_masks_exclude = (1.0, 80.0)
    weights_mask_exclude = (1.0, 300.0)  # for LUVAR data
    #weights_mask_exclude = (1.0, 361.0)  # for DLCST data

    def __init__(self, is_mask_exclude: bool = False) -> None:
        if is_mask_exclude:
            self._weights = self.weights_mask_exclude
        else:
            self._weights = self.weights_no_masks_exclude
        super(WeightedBinaryCrossEntropyFixedWeights, self).__init__(is_mask_exclude)
        self._name_fun_out = 'weight_bin_cross_fixed'

    def _get_weights(self, y_true: np.ndarray) -> Tuple[float, float]:
        return self._weights


class BinaryCrossEntropyFocalLoss(MetricBase):
    # Binary cross entropy + Focal loss
    _gamma_default = 2.0

    def __init__(self, gamma: float = _gamma_default, is_mask_exclude: bool = False) -> None:
        self._gamma = gamma
        super(BinaryCrossEntropyFocalLoss, self).__init__(is_mask_exclude)
        self._name_fun_out = 'bin_cross_focal_loss'

    def _get_predprobs_classes(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prob_1 = np.where(y_true == 1.0, y_pred, 1)
        prob_0 = np.where(y_true == 0.0, y_pred, 0)
        return (prob_1, prob_0)

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(- y_true * pow(1.0 - y_pred, self._gamma) * np.log(y_pred + _EPS)
                       - (1.0 - y_true) * pow(y_pred, self._gamma) * np.log(1.0 - y_pred + _EPS))

    def _compute_masked(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        mask = self._get_mask(y_true)
        return np.mean((- y_true * pow(1.0 - y_pred, self._gamma) * np.log(y_pred + _EPS)
                        - (1.0 - y_true) * pow(y_pred, self._gamma) * np.log(1.0 - y_pred + _EPS)) * mask)


class DiceCoefficient(MetricBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(DiceCoefficient, self).__init__(is_mask_exclude)
        self._name_fun_out = 'dice'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (2.0 * np.sum(y_true * y_pred)) / (np.sum(y_true) + np.sum(y_pred) + _SMOOTH)


class TruePositiveRate(MetricBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(TruePositiveRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'tp_rate'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum(y_true * y_pred) / (np.sum(y_true) + _SMOOTH)


class TrueNegativeRate(MetricBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(TrueNegativeRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'tn_rate'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum((1.0 - y_true) * (1.0 - y_pred)) / (np.sum((1.0 - y_true)) + _SMOOTH)


class FalsePositiveRate(MetricBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(FalsePositiveRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'fp_rate'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum((1.0 - y_true) * y_pred) / (np.sum((1.0 - y_true)) + _SMOOTH)


class FalseNegativeRate(MetricBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(FalseNegativeRate, self).__init__(is_mask_exclude)
        self._name_fun_out = 'fn_rate'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum(y_true * (1.0 - y_pred)) / (np.sum(y_true) + _SMOOTH)


class AirwayCompleteness(MetricBase):
    _is_use_ytrue_cenlines = True
    _is_use_ypred_cenlines = False

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayCompleteness, self).__init__(is_mask_exclude)
        self._name_fun_out = 'completeness'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum(y_true * y_pred) / (np.sum(y_true) + _SMOOTH)


class AirwayVolumeLeakage(MetricBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayVolumeLeakage, self).__init__(is_mask_exclude)
        self._name_fun_out = 'volume_leakage'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum((1.0 - y_true) * y_pred) / (np.sum(y_pred) + _SMOOTH)


class AirwayCentrelineLeakage(MetricBase):
    _is_use_ytrue_cenlines = False
    _is_use_ypred_cenlines = True

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayCentrelineLeakage, self).__init__(is_mask_exclude)
        self._name_fun_out = 'cenline_leakage'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum((1.0 - y_true) * y_pred) / (np.sum(y_pred) + _SMOOTH)


class AirwayTreeLength(MetricBase):
    _is_use_ytrue_cenlines = False
    _is_use_ypred_cenlines = True
    _is_use_img_voxel_size = True

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayTreeLength, self).__init__(is_mask_exclude)
        self._name_fun_out = 'tree_length'

    def _get_voxel_diag(self) -> np.ndarray:
        return np.sqrt(self._voxel_size.dot(self._voxel_size))

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum(y_pred) * self._get_voxel_diag()


class AirwayCentrelineDistanceFalsePositiveError(MetricBase):
    _is_use_ytrue_cenlines = True
    _is_use_ypred_cenlines = True
    _is_use_img_voxel_size = True

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayCentrelineDistanceFalsePositiveError, self).__init__(is_mask_exclude)
        self._name_fun_out = 'cenline_dist_fp_err'

    def _get_cenline_coords(self, y_input: np.ndarray) -> np.ndarray:
        return np.asarray(np.argwhere(y_input > 0)) * self._voxel_size

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = self._get_cenline_coords(y_true)
        y_pred = self._get_cenline_coords(y_pred)
        dist_y = distance.cdist(y_pred, y_true)
        return np.mean(np.min(dist_y, axis=1))


class AirwayCentrelineDistanceFalseNegativeError(MetricBase):
    _is_use_ytrue_cenlines = True
    _is_use_ypred_cenlines = True
    _is_use_img_voxel_size = True

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayCentrelineDistanceFalseNegativeError, self).__init__(is_mask_exclude)
        self._name_fun_out = 'cenline_dist_fn_err'

    def _get_cenline_coords(self, y_input: np.ndarray) -> np.ndarray:
        return np.asarray(np.argwhere(y_input > 0)) * self._voxel_size

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = self._get_cenline_coords(y_true)
        y_pred = self._get_cenline_coords(y_pred)
        dist_y = distance.cdist(y_pred, y_true)
        return np.mean(np.min(dist_y, axis=0))



# ******************** TO IMPLEMENT BY ALEX ********************

class MetricModifiedBase(MetricBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MetricModifiedBase, self).__init__(is_mask_exclude)

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if self._is_mask_exclude:
            return self._compute_masked(y_true, y_pred)
        else:
            return self._compute(y_true, y_pred)

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _compute_masked(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self._compute(self._get_masked_input(y_true, y_true),
                             self._get_masked_input(y_pred, y_true))

    def _get_mask(self, y_true: np.ndarray) -> np.ndarray:
        return np.where(y_true == self._value_mask_exclude, 0, 1)

    def _get_masked_input(self, y_input: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.where(y_true == self._value_mask_exclude, 0, y_input)


# Signal-to-Noise ratio
class SNR(MetricModifiedBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(SNR, self).__init__(is_mask_exclude)
        self._name_fun_out = 'snr'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        mean = np.mean(y_pred.flatten())
        std = np.std(y_pred.flatten())
        return mean / std
        #return signaltonoise(y_pred, axis=None)


# Peak Signal-to-Noise ratio
class PSNR(MetricModifiedBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(PSNR, self).__init__(is_mask_exclude)
        self._name_fun_out = 'psnr'
        self._max_value = 255.0

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        mse = np.mean(np.square(y_pred.flatten() - y_true.flatten()))
        if mse == 0.0:
            return 0.0
        else:
            return 20.0 * np.log10(self._max_value / np.sqrt(mse))


# SSIM: Structural Similarity Index
class SSIM(MetricModifiedBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(SSIM, self).__init__(is_mask_exclude)
        self._name_fun_out = 'ssim'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return ssim(y_true, y_pred, data_range=255.0)


# SSIM HOMEMADE FOR TESTING
class SSIM_Homemade(MetricModifiedBase):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(SSIM_Homemade, self).__init__(is_mask_exclude)
        self._name_fun_out = 'ssim_homemade'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        cons_K1 = 0.01
        cons_K2 = 0.03
        range_L = 255.0
        cons_C1 = (cons_K1*range_L)**2
        cons_C2 = (cons_K2*range_L)**2
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        mean_y_true = np.mean(y_true, axis=-1)
        mean_y_pred = np.mean(y_pred, axis=-1)
        std_y_true = np.sqrt(np.mean(np.square(y_true - mean_y_true), axis=-1))
        std_y_pred = np.sqrt(np.mean(np.square(y_pred - mean_y_pred), axis=-1))
        std_y_true_pred = np.mean((y_true - mean_y_true) * (y_pred - mean_y_pred), axis=-1)
        return (2*mean_y_true*mean_y_pred + cons_C1) / (mean_y_true**2 + mean_y_pred**2 + cons_C1) *\
               (2*std_y_true_pred + cons_C2) / (std_y_true**2 + std_y_pred**2 + cons_C2)

    def _compute_masked(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        cons_K1 = 0.01
        cons_K2 = 0.03
        range_L = 255.0
        cons_C1 = (cons_K1*range_L)**2
        cons_C2 = (cons_K2*range_L)**2
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        mask = self._get_mask(y_true)
        mean_y_true = np.mean(y_true * mask, axis=-1)
        mean_y_pred = np.mean(y_pred * mask, axis=-1)
        std_y_true = np.sqrt(np.mean(np.square(y_true - mean_y_true) * mask, axis=-1))
        std_y_pred = np.sqrt(np.mean(np.square(y_pred - mean_y_pred) * mask, axis=-1))
        std_y_true_pred = np.mean((y_true - mean_y_true) * (y_pred - mean_y_pred) * mask, axis=-1)
        return (2*mean_y_true*mean_y_pred + cons_C1) / (mean_y_true**2 + mean_y_pred**2 + cons_C1) *\
               (2*std_y_true_pred + cons_C2) / (std_y_true**2 + std_y_pred**2 + cons_C2)
