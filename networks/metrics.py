
from typing import Tuple
import numpy as np
from scipy.spatial import distance

_eps = 1.0e-7
_smooth = 1.0


class Metrics(object):
    _max_size_memory_safe = 5e+08
    _value_mask_exclude = -1
    _is_use_ytrue_cenlines = False
    _is_use_ypred_cenlines = False

    def __init__(self, is_mask_exclude: bool = False) -> None:
        self._is_mask_exclude = is_mask_exclude
        self._name_func_out = None

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


class MeanSquaredError(Metrics):

    def __init__(self, is_masks_exclude: bool = False) -> None:
        super(MeanSquaredError, self).__init__(is_masks_exclude)
        self._name_func_out  = 'mean_squared'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.square(y_pred - y_true))

    def _compute_masked(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        mask = self._get_mask(y_true)
        return np.mean(np.square(y_pred - y_true) * mask)


class MeanSquaredErrorLogarithmic(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(MeanSquaredErrorLogarithmic, self).__init__(is_mask_exclude)
        self._name_func_out  = 'mean_squared_log'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.square(np.log(np.clip(y_pred, _eps, None) + 1.0) -
                                 np.log(np.clip(y_true, _eps, None) + 1.0)))

    def _compute_masked(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        mask = self._get_mask(y_true)
        return np.mean(np.square(np.log(np.clip(y_pred, _eps, None) + 1.0) -
                                 np.log(np.clip(y_true, _eps, None) + 1.0)) * mask)


class BinaryCrossEntropy(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(BinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_func_out = 'bin_cross'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(- y_true * np.log(y_pred + _eps)
                       - (1.0 - y_true) * np.log(1.0 - y_pred + _eps))

    def _compute_masked(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        mask = self._get_mask(y_true)
        return np.mean((- y_true * np.log(y_pred + _eps)
                        - (1.0 - y_true) * np.log(1.0 - y_pred + _eps)) * mask)


class WeightedBinaryCrossEntropy(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(WeightedBinaryCrossEntropy, self).__init__(is_mask_exclude)
        self._name_func_out = 'weight_bin_cross'

    def _get_weights(self, y_true: np.ndarray) -> Tuple[float, float]:
        num_class_1 = np.count_nonzero(y_true == 1)
        num_class_0 = np.count_nonzero(y_true == 0)
        return (1.0, num_class_0 / (float(num_class_1) +_eps))

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        weights = self._get_weights(y_true)
        return np.mean(- weights[1] * y_true * np.log(y_pred + _eps)
                       - weights[0] * (1.0 - y_true) * np.log(1.0 - y_pred + _eps))

    def _compute_masked(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        weights = self._get_weights(y_true)
        mask = self._get_mask(y_true)
        return np.mean((- weights[1] * y_true * np.log(y_pred + _eps)
                        - weights[0] * (1.0 - y_true) * np.log(1.0 - y_pred + _eps)) * mask)


class WeightedBinaryCrossEntropyFixedWeights(Metrics):
    weights_no_masks_exclude = [1.0, 80.0]
    weights_mask_exclude = [1.0, 300.0]  # for LUVAR data
    #weights_mask_exclude = [1.0, 361.0]  # for DLCST data

    def __init__(self, is_mask_exclude: bool = False) -> None:
        if is_mask_exclude:
            self._weights = self.weights_mask_exclude
        else:
            self._weights = self.weights_no_masks_exclude
        super(WeightedBinaryCrossEntropyFixedWeights, self).__init__(is_mask_exclude)
        self._name_func_out = 'weight_bin_cross_fixed'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(- self._weights[1] * y_true * np.log(y_pred + _eps)
                       - self._weights[0] * (1.0 - y_true) * np.log(1.0 - y_pred + _eps))

    def _compute_masked(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        mask = self._get_mask(y_true)
        return np.mean((- self._weights[1] * y_true * np.log(y_pred + _eps)
                        - self._weights[0] * (1.0 - y_true) * np.log(1.0 - y_pred + _eps)) * mask)


class BinaryCrossEntropyFocalLoss(Metrics):
    # Binary cross entropy + Focal loss
    _gamma_default = 2.0

    def __init__(self, gamma: float = _gamma_default, is_mask_exclude: bool = False) -> None:
        self._gamma = gamma
        super(BinaryCrossEntropyFocalLoss, self).__init__(is_mask_exclude)
        self._name_func_out = 'bin_cross_focal_loss'

    def _get_predprobs_classes(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prob_1 = np.where(y_true == 1.0, y_pred, 1)
        prob_0 = np.where(y_true == 0.0, y_pred, 0)
        return (prob_1, prob_0)

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(- y_true * pow(1.0 - y_pred, self._gamma) * np.log(y_pred + _eps)
                       - (1.0 - y_true) * pow(y_pred, self._gamma) * np.log(1.0 - y_pred + _eps))

    def _compute_masked(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        mask = self._get_mask(y_true)
        return np.mean((- y_true * pow(1.0 - y_pred, self._gamma) * np.log(y_pred + _eps)
                        - (1.0 - y_true) * pow(y_pred, self._gamma) * np.log(1.0 - y_pred + _eps)) * mask)


class DiceCoefficient(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(DiceCoefficient, self).__init__(is_mask_exclude)
        self._name_func_out = 'dice'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (2.0 * np.sum(y_true * y_pred)) / (np.sum(y_true) + np.sum(y_pred) + _smooth)


class TruePositiveRate(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(TruePositiveRate, self).__init__(is_mask_exclude)
        self._name_func_out = 'tpr'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum(y_true * y_pred) / (np.sum(y_true) + _smooth)


class TrueNegativeRate(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(TrueNegativeRate, self).__init__(is_mask_exclude)
        self._name_func_out = 'tnr'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum((1.0 - y_true) * (1.0 - y_pred)) / (np.sum((1.0 - y_true)) + _smooth)


class FalsePositiveRate(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(FalsePositiveRate, self).__init__(is_mask_exclude)
        self._name_func_out = 'fpr'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum((1.0 - y_true) * y_pred) / (np.sum((1.0 - y_true)) + _smooth)


class FalseNegativeRate(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(FalseNegativeRate, self).__init__(is_mask_exclude)
        self._name_func_out = 'fnr'

    def _compute(self, y_true, y_pred):
        return np.sum(y_true * (1.0 - y_pred)) / (np.sum(y_true) + _smooth)


class AirwayCompleteness(Metrics):
    _is_use_ytrue_cenlines = True
    _is_use_ypred_cenlines = False

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayCompleteness, self).__init__(is_mask_exclude)
        self._name_func_out = 'completeness'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum(y_true * y_pred) / (np.sum(y_true) + _smooth)


class AirwayVolumeLeakage(Metrics):

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayVolumeLeakage, self).__init__(is_mask_exclude)
        self._name_func_out = 'volume_leakage'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum((1.0 - y_true) * y_pred) / (np.sum(y_pred) + _smooth)


class AirwayCentrelineLeakage(Metrics):
    _is_use_ytrue_cenlines = False
    _is_use_ypred_cenlines = True

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayCentrelineLeakage, self).__init__(is_mask_exclude)
        self._name_func_out = 'centreline_leakage'

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum((1.0 - y_true) * y_pred) / (np.sum(y_pred) + _smooth)


class AirwayCentrelineDistanceFalsePositiveError(Metrics):
    _is_use_ytrue_cenlines = True
    _is_use_ypred_cenlines = True

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayCentrelineDistanceFalsePositiveError, self).__init__(is_mask_exclude)
        self._name_func_out = 'centreline_dist_FP_error'

    @staticmethod
    def _get_voxel_scaling(y_input: np.ndarray) -> np.ndarray:
        #return np.diag(y_input.affine)[:3]
        return np.asarray([1.0, 1.0, 1.0])

    @classmethod
    def _get_centreline_coords(cls, y_input: np.ndarray) -> np.ndarray:
        return np.asarray(np.argwhere(y_input > 0)) * cls._get_voxel_scaling(y_input)

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = self._get_centreline_coords(y_true)
        y_pred = self._get_centreline_coords(y_pred)
        dist_y = distance.cdist(y_pred, y_true)
        return np.mean(np.min(dist_y, axis=1))


class AirwayCentrelineDistanceFalseNegativeError(Metrics):
    _is_use_ytrue_cenlines = True
    _is_use_ypred_cenlines = True

    def __init__(self, is_mask_exclude: bool = False) -> None:
        super(AirwayCentrelineDistanceFalseNegativeError, self).__init__(is_mask_exclude)
        self._name_func_out = 'centreline_dist_FN_error'

    @staticmethod
    def _get_voxel_scaling(y_input: np.ndarray) -> np.ndarray:
        #return np.diag(y_input.affine)[:3]
        return np.asarray([1.0, 1.0, 1.0])

    @classmethod
    def _get_centreline_coords(cls, y_input: np.ndarray) -> np.ndarray:
        return np.asarray(np.argwhere(y_input > 0)) * cls._get_voxel_scaling(y_input)

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = self._get_centreline_coords(y_true)
        y_pred = self._get_centreline_coords(y_pred)
        dist_y = distance.cdist(y_pred, y_true)
        return np.mean(np.min(dist_y, axis=0))