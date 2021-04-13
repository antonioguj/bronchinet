
from typing import Tuple, List
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import pandas as pd

from common.constant import TYPE_DNNLIB_USED
from common.exceptionmanager import catch_error_exception
if TYPE_DNNLIB_USED == 'Keras':
    from models.keras.metrics import AirwayCompleteness, AirwayVolumeLeakage, DiceCoefficient
elif TYPE_DNNLIB_USED == 'Pytorch':
    from models.pytorch.metrics import AirwayCompleteness, AirwayVolumeLeakage, DiceCoefficient

matplotlib.use('Agg')


def compute_assignment(list_detections: List[np.ndarray],
                       list_groundtruth: List[np.ndarray],
                       allowed_distance: float
                       ) -> Tuple[int, int, int]:
    # the assignment is based on the hungarian algorithm

    # build cost matrix
    cost_matrix = np.zeros([len(list_groundtruth), len(list_detections)])
    for i, point_r1 in enumerate(list_groundtruth):
        for j, point_r2 in enumerate(list_detections):
            cost_matrix[i, j] = euclidean(point_r1, point_r2)

    # perform assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # threshold points too far
    row_ind_thresholded = []
    col_ind_thresholded = []
    for i in range(len(row_ind)):
        if cost_matrix[row_ind[i], col_ind[i]] < allowed_distance:
            row_ind_thresholded.append(row_ind[i])
            col_ind_thresholded.append(col_ind[i])

    # compute stats
    num_p = len(list_groundtruth)
    num_tp = len(row_ind_thresholded)
    num_fp = len(list_detections) - num_tp

    return (num_p, num_tp, num_fp)


def compute_froc_from_lists_matrix(list_ids: List[int],
                                   list_detections: List[np.ndarray],
                                   list_groundtruth: List[np.ndarray],
                                   allowed_distance: float
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    # list_detection: first dimension: number of images
    # list_groundtruth: first dimension: number of images

    # get maximum number of detection per image across the dataset
    max_nbr_detections = 0
    for detections in list_detections:
        if len(detections) > max_nbr_detections:
            max_nbr_detections = len(detections)

    sensitivity_matrix = pd.DataFrame(columns=list_ids)
    matrix_fps = pd.DataFrame(columns=list_ids)

    for i in range(1, max_nbr_detections):
        sensitivity_per_image = {}
        num_fp_per_image = {}
        for image_nbr, groundtruth in enumerate(list_groundtruth):
            image_id = list_ids[image_nbr]
            if len(groundtruth) > 0:  # check that ground truth contains at least one annotation
                if i <= len(list_detections[image_nbr]):  # if there are detections
                    # compute P, TP, FP per image
                    detections = list_detections[image_nbr][-i]
                    (num_p, num_tp, num_fp) = compute_assignment(detections, groundtruth, allowed_distance)
                else:
                    num_p = len(groundtruth)
                    num_tp, num_fp = 0, 0

                # append results to list
                num_fp_per_image[image_id] = num_fp
                sensitivity_per_image[image_id] = num_tp * 1. / num_p

            elif len(groundtruth) == 0 and i <= len(list_detections[image_nbr]):  # if no annotations but detections
                num_fp = len(list_detections[image_nbr][-i])
                num_fp_per_image[image_id] = num_fp
                sensitivity_per_image[image_id] = None

        sensitivity_matrix = sensitivity_matrix.append(sensitivity_per_image, ignore_index=True)
        matrix_fps = matrix_fps.append(num_fp_per_image, ignore_index=True)

    return (sensitivity_matrix, matrix_fps)


def compute_froc_from_lists(list_detections: List[np.ndarray],
                            list_groundtruth: List[np.ndarray],
                            allowed_distance: float
                            ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    # get maximum number of detection per image across the dataset
    max_nbr_detections = 0
    for detections in list_detections:
        if len(detections) > max_nbr_detections:
            max_nbr_detections = len(detections)

    list_sensitivity = []
    list_avg_fp = []
    list_sensitivity_std = []
    list_avg_fp_std = []

    for i in range(max_nbr_detections):
        list_sensitivity_per_image = []
        list_num_fp_per_image = []
        for image_nbr, groundtruth in enumerate(list_groundtruth):
            if len(groundtruth) > 0:  # check that ground truth contains at least one annotation
                if i <= len(list_detections[image_nbr]):  # if there are detections
                    # compute P, TP, FP per image
                    detections = list_detections[image_nbr][-i]
                    (num_p, num_tp, num_fp) = compute_assignment(detections, groundtruth, allowed_distance)
                else:
                    num_p = len(groundtruth)
                    num_tp, num_fp = 0, 0

                # append results to list
                list_num_fp_per_image.append(num_fp)
                list_sensitivity_per_image.append(num_tp * 1. / num_p)

            elif len(groundtruth) == 0 and i <= len(list_detections[image_nbr]):  # if no annotations but detections
                num_fp = len(list_detections[image_nbr][-i])
                list_num_fp_per_image.append(num_fp)
                list_sensitivity_per_image.append(None)

                # average sensitivity and FP over the proba map, for a given threshold
        list_sensitivity.append(np.mean(list_sensitivity_per_image))
        list_avg_fp.append(np.mean(list_num_fp_per_image))
        list_sensitivity_std.append(np.std(list_sensitivity_per_image))
        list_avg_fp_std.append(np.std(list_num_fp_per_image))

    return (list_sensitivity, list_avg_fp, list_sensitivity_std, list_avg_fp_std)


def compute_confusion_matrix_elements(thresholded_proba_map: np.ndarray,
                                      groundtruth: np.ndarray,
                                      allowed_distance: float
                                      ) -> Tuple[int, int, int]:
    if allowed_distance == 0 and type(groundtruth) == np.ndarray:
        num_p = np.count_nonzero(groundtruth)
        num_tp = np.count_nonzero(thresholded_proba_map * groundtruth)
        num_fp = np.count_nonzero(thresholded_proba_map - (thresholded_proba_map * groundtruth))
    else:

        # reformat ground truth to a list
        if type(groundtruth) == np.ndarray:
            # convert ground truth binary map to list of coordinates
            labels, num_features = ndimage.label(groundtruth)
            list_groundtruth = ndimage.measurements.center_of_mass(groundtruth, labels, range(1, num_features + 1))
        elif type(groundtruth) == list:
            list_groundtruth = groundtruth
        else:
            message = 'groundtruth should be either of type list or np.ndarray and is of type %s' % (type(groundtruth))
            catch_error_exception(message)

        # reformat thresholded_proba_map to a list
        labels, num_features = ndimage.label(thresholded_proba_map)
        list_proba_map = ndimage.measurements.center_of_mass(thresholded_proba_map, labels, range(1, num_features + 1))

        # compute P, TP and FP
        (num_p, num_tp, num_fp) = compute_assignment(list_proba_map, list_groundtruth, allowed_distance)

    return (num_p, num_tp, num_fp)


def compute_froc(proba_map: np.ndarray,
                 groundtruth: np.ndarray,
                 allowed_distance: float,
                 list_threshold: List[float] = None
                 ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # INPUTS :
    # proba_map : numpy array of dimension [number of image, xdim, ydim,...], values preferably in [0,1]
    # groundtruth: numpy array of dimension [number of image, xdim, ydim,...], values in {0,1}; or list of coordinates
    # allowed_distance: Integer euclidian distance distance in pixels to consider a detection as valid
    # nbr_of_thresholds: Integer number of thresholds to compute to plot the FROC
    # range_threshold: list of 2 floats. Beginning and end of the range of thresholds with which to plot the FROC
    # OUTPUTS :
    # list_sensitivity_treshold: list of average sensitivity over the set of images for increasing thresholds
    # list_FPavg_treshold: list of average FP over the set of images for increasing thresholds
    # list_threshold: list of thresholds

    # rescale ground truth and proba map between 0 and 1
    proba_map = proba_map.astype(np.float32)
    proba_map = (proba_map - np.min(proba_map)) / (np.max(proba_map) - np.min(proba_map))
    if type(groundtruth) == np.ndarray:
        # verify that proba_map and groundtruth have the same shape
        if proba_map.shape != groundtruth.shape:
            raise ValueError('Error. Proba map and ground truth have different shapes.')

        groundtruth = groundtruth.astype(np.float32)
        groundtruth = (groundtruth - np.min(groundtruth)) / (np.max(groundtruth) - np.min(groundtruth))

    # define the thresholds
    if list_threshold is None:
        nbr_thresholds = 10
        list_threshold = (np.linspace(np.min(proba_map), np.max(proba_map), nbr_thresholds)).tolist()

    list_sensitivity_treshold = []
    list_avg_fp_treshold = []
    # loop over thresholds
    for threshold in list_threshold:
        sensitivity_list_proba_map = []
        list_fp_proba_map = []
        # loop over proba map
        for i in range(len(proba_map)):
            # threshold the proba map
            thresholded_proba_map = np.zeros(np.shape(proba_map[i]))
            thresholded_proba_map[proba_map[i] >= threshold] = 1

            # save proba maps
            # imageio.imwrite('thresholded_proba_map_'+str(threshold)+'.png', thresholded_proba_map)

            # compute P, TP, and FP for this threshold and this proba map
            (num_p, num_tp, num_fp) = compute_confusion_matrix_elements(thresholded_proba_map,
                                                                        groundtruth[i], allowed_distance)

            # append results to list
            list_fp_proba_map.append(num_fp)
            # check that ground truth contains at least one positive
            if (type(groundtruth[i]) == np.ndarray and np.count_nonzero(groundtruth[i]) > 0) or \
                    (type(groundtruth[i]) == list and len(groundtruth[i]) > 0):
                sensitivity_list_proba_map.append(num_tp * 1. / num_p)

        # average sensitivity and FP over the proba map, for a given threshold
        list_sensitivity_treshold.append(np.mean(sensitivity_list_proba_map))
        list_avg_fp_treshold.append(np.mean(list_fp_proba_map))

    return (list_sensitivity_treshold, list_avg_fp_treshold)


def compute_roc_completeness_leakage(pred_probmaps: np.ndarray,
                                     groundtruth: np.ndarray,
                                     centrelines: np.ndarray,
                                     list_threshold: List[float] = None
                                     ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    # rescale ground truth and proba map between 0 and 1
    pred_probmaps = pred_probmaps.astype(np.float32)
    pred_probmaps = (pred_probmaps - np.min(pred_probmaps)) / (np.max(pred_probmaps) - np.min(pred_probmaps))

    # define the thresholds
    if list_threshold is None:
        nbr_thresholds = 10
        list_threshold = (np.linspace(np.min(pred_probmaps), np.max(pred_probmaps), nbr_thresholds)).tolist()

    list_completeness_threshold = []
    list_volumeleakage_threshold = []
    list_dice_coeff_threshold = []

    # loop over thresholds
    for threshold in list_threshold:
        list_completeness_probmap = []
        list_volumeleakage_probmap = []
        list_dice_coeff_probmap = []

        # loop over proba map
        for i in range(len(pred_probmaps)):
            # threshold the proba map
            thresholded_probmaps = np.zeros(np.shape(pred_probmaps[i]))
            thresholded_probmaps[pred_probmaps[i] >= threshold] = 1

            # compute Completeness
            completeness_value = AirwayCompleteness().compute_np(centrelines, thresholded_probmaps)
            # compute Volume Leakage
            volume_leakage_value = AirwayVolumeLeakage().compute_np(groundtruth, thresholded_probmaps)
            # compute Dice Coefficient
            dice_coeff_value = DiceCoefficient().compute_np(groundtruth, thresholded_probmaps)

            # append results to list
            list_volumeleakage_probmap.append(volume_leakage_value)
            list_dice_coeff_probmap.append(dice_coeff_value)

            # check that ground truth contains at least one positive
            if (type(groundtruth[i]) == np.ndarray and np.count_nonzero(groundtruth[i]) > 0) or \
                    (type(groundtruth[i]) == list and len(groundtruth[i]) > 0):
                list_completeness_probmap.append(completeness_value)

        # average sensitivity and FP over the proba map, for a given threshold
        list_completeness_threshold.append(np.mean(list_completeness_probmap))
        list_volumeleakage_threshold.append(np.mean(list_volumeleakage_probmap))
        list_dice_coeff_threshold.append(np.mean(list_dice_coeff_probmap))

    return (list_completeness_threshold, list_volumeleakage_threshold, list_dice_coeff_threshold)


def plot_froc(in_xdata: np.ndarray,
              in_ydata: np.ndarray,
              save_path: str = None,
              list_threshold: List[float] = None
              ) -> None:
    plt.figure()
    plt.plot(in_xdata, in_ydata, 'o-')
    plt.xlabel('FPavg')
    plt.ylabel('Sensitivity')

    # annotate thresholds
    if list_threshold is not None:
        # round thresholds
        list_threshold = ['%.2f' % elem for elem in list_threshold]
        xy_buffer = None
        for i, xy in enumerate(zip(in_xdata, in_ydata)):
            if xy != xy_buffer:
                plt.annotate(str(list_threshold[i]), xy=xy, textcoords='data')
                xy_buffer = xy

    if save_path is not None:
        print("saving figure...")
        plt.savefig(save_path)
        plt.close()
    else:
        print("plotting figure...")
        plt.show()
