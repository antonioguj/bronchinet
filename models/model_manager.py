
from typing import List, Tuple

from common.constant import TYPE_DNNLIB_USED
from common.exceptionmanager import catch_error_exception
if TYPE_DNNLIB_USED == 'Pytorch':
    from models.pytorch.callbacks import RecordLossHistory
    from models.pytorch.metrics import Metric as Metric_train, \
                                       CombineTwoMetrics as CombineTwoMetrics_train, \
                                       MeanSquaredError as MeanSquaredError_train, \
                                       MeanSquaredErrorLogarithmic as MeanSquaredErrorLogarithmic_train, \
                                       BinaryCrossEntropy as BinaryCrossEntropy_train, \
                                       WeightedBinaryCrossEntropy as WeightedBinaryCrossEntropy_train, \
                                       WeightedBinaryCrossEntropyFixedWeights as WeightedBinaryCrossEntropyFixedWeights_train, \
                                       BinaryCrossEntropyFocalLoss as BinaryCrossEntropyFocalLoss_train, \
                                       DiceCoefficient as DiceCoefficient_train, \
                                       TruePositiveRate as TruePositiveRate_train, \
                                       TrueNegativeRate as TrueNegativeRate_train, \
                                       FalsePositiveRate as FalsePositiveRate_train, \
                                       FalseNegativeRate as FalseNegativeRate_train, \
                                       AirwayCompleteness as AirwayCompleteness_train, \
                                       AirwayVolumeLeakage as AirwayVolumeLeakage_train, \
                                       AirwayCentrelineLeakage as AirwayCentrelineLeakage_train, \
                                       LIST_AVAIL_METRICS as LIST_AVAIL_METRICS_TRAIN
    from models.pytorch.networks import UNet3D_Original, UNet3D_General, UNet3D_Plugin, LIST_AVAIL_NETWORKS
    from models.pytorch.optimizers import SGD, SGD_mom, RMSprop, Adagrad, Adadelta, Adam, LIST_AVAIL_OPTIMIZERS
    from models.pytorch.visualmodelparams import VisualModelParams
elif TYPE_DNNLIB_USED == 'Keras':
    from models.keras.callbacks import RecordLossHistory, EarlyStopping, ModelCheckpoint
    from models.keras.metrics import Metric as Metric_train, \
                                     CombineTwoMetrics as CombineTwoMetrics_train, \
                                     MeanSquaredError as MeanSquaredError_train, \
                                     MeanSquaredErrorLogarithmic as MeanSquaredErrorLogarithmic_train, \
                                     BinaryCrossEntropy as BinaryCrossEntropy_train, \
                                     WeightedBinaryCrossEntropy as WeightedBinaryCrossEntropy_train, \
                                     WeightedBinaryCrossEntropyFixedWeights as WeightedBinaryCrossEntropyFixedWeights_train, \
                                     BinaryCrossEntropyFocalLoss as BinaryCrossEntropyFocalLoss_train, \
                                     DiceCoefficient as DiceCoefficient_train, \
                                     TruePositiveRate as TruePositiveRate_train, \
                                     TrueNegativeRate as TrueNegativeRate_train, \
                                     FalsePositiveRate as FalsePositiveRate_train, \
                                     FalseNegativeRate as FalseNegativeRate_train, \
                                     AirwayCompleteness as AirwayCompleteness_train, \
                                     AirwayVolumeLeakage as AirwayVolumeLeakage_train, \
                                     AirwayCentrelineLeakage as AirwayCentrelineLeakage_train, \
                                     LIST_AVAIL_METRICS as LIST_AVAIL_METRICS_TRAIN
    from models.keras.networks import UNet3D_Original, UNet3D_General, UNet3D_Plugin, LIST_AVAIL_NETWORKS
    from models.keras.optimizers import SGD, SGD_mom, RMSprop, Adagrad, Adadelta, Adam, LIST_AVAIL_OPTIMIZERS
    from models.keras.visualmodelparams import VisualModelParams
from models.metrics import MetricBase, MeanSquaredError, MeanSquaredErrorLogarithmic, BinaryCrossEntropy, \
                           WeightedBinaryCrossEntropy, WeightedBinaryCrossEntropyFixedWeights, DiceCoefficient, \
                           TruePositiveRate, TrueNegativeRate, FalsePositiveRate, FalseNegativeRate, \
                           AirwayCompleteness, AirwayVolumeLeakage, AirwayCentrelineLeakage, \
                           AirwayCentrelineDistanceFalseNegativeError, AirwayCentrelineDistanceFalsePositiveError, \
                           LIST_AVAIL_METRICS
from models.networks import ConvNetBase



def get_metric(type_metric: str,
               is_mask_exclude: bool = False,
               **kwargs) -> MetricBase:
    if type_metric == 'MeanSquaredError':
        return MeanSquaredError(is_mask_exclude=is_mask_exclude)
    if type_metric == 'MeanSquaredErrorLogarithmic':
        return MeanSquaredErrorLogarithmic(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'BinaryCrossEntropy':
        return BinaryCrossEntropy(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'WeightedBinaryCrossEntropy':
        return WeightedBinaryCrossEntropy(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'WeightedBinaryCrossEntropyFixedWeights':
        return WeightedBinaryCrossEntropyFixedWeights(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'DiceCoefficient':
        return DiceCoefficient(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'TruePositiveRate':
        return TruePositiveRate(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'TrueNegativeRate':
        return TrueNegativeRate(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'FalsePositiveRate':
        return FalsePositiveRate(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'FalseNegativeRate':
        return FalseNegativeRate(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'AirwayCompleteness':
        return AirwayCompleteness(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'AirwayVolumeLeakage':
        return AirwayVolumeLeakage(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'AirwayCentrelineLeakage':
        return AirwayCentrelineLeakage(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'AirwayCentrelineDistanceFalsePositiveError':
        return AirwayCentrelineDistanceFalsePositiveError(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'AirwayCentrelineDistanceFalseNegativeError':
        return AirwayCentrelineDistanceFalseNegativeError(is_mask_exclude=is_mask_exclude)
    else:
        message = 'Choice Metric not found. Metrics available: %s' % (', '.join(LIST_AVAIL_METRICS))
        catch_error_exception(message)


def get_metric_train(type_metric: str,
                     is_mask_exclude: bool = False,
                     **kwargs) -> Metric_train:
    is_combine_metrics = kwargs['is_combine_metrics'] if 'is_combine_metrics' in kwargs.keys() else False
    if is_combine_metrics:
        type_metrics_1, type_metrics_2 = type_metric.split('_')
        weights_metrics = kwargs['weights_metrics'] if 'weights_metrics' in kwargs.keys() else (1.0, 1.0)
        metrics_1 = get_metric_train(type_metrics_1, is_mask_exclude)
        metrics_2 = get_metric_train(type_metrics_2, is_mask_exclude)
        return CombineTwoMetrics_train(metrics_1, metrics_2, weights_metrics)
    else:
        if type_metric == 'MeanSquaredError':
            return MeanSquaredError_train(is_mask_exclude=is_mask_exclude)
        if type_metric == 'MeanSquaredErrorLogarithmic':
            return MeanSquaredErrorLogarithmic_train(is_mask_exclude=is_mask_exclude)
        elif type_metric == 'BinaryCrossEntropy':
            return BinaryCrossEntropy_train(is_mask_exclude=is_mask_exclude)
        elif type_metric == 'WeightedBinaryCrossEntropy':
            return WeightedBinaryCrossEntropy_train(is_mask_exclude=is_mask_exclude)
        elif type_metric == 'WeightedBinaryCrossEntropyFixedWeights':
            return WeightedBinaryCrossEntropyFixedWeights_train(is_mask_exclude=is_mask_exclude)
        elif type_metric == 'BinaryCrossEntropyFocalLoss':
            return BinaryCrossEntropyFocalLoss_train(is_mask_exclude=is_mask_exclude)
        elif type_metric == 'DiceCoefficient':
            return DiceCoefficient_train(is_mask_exclude=is_mask_exclude)
        elif type_metric == 'TruePositiveRate':
            return TruePositiveRate_train(is_mask_exclude=is_mask_exclude)
        elif type_metric == 'TrueNegativeRate':
            return TrueNegativeRate_train(is_mask_exclude=is_mask_exclude)
        elif type_metric == 'FalsePositiveRate':
            return FalsePositiveRate_train(is_mask_exclude=is_mask_exclude)
        elif type_metric == 'FalseNegativeRate':
            return FalseNegativeRate_train(is_mask_exclude=is_mask_exclude)
        elif type_metric == 'AirwayCompleteness':
            return AirwayCompleteness_train(is_mask_exclude=is_mask_exclude)
        elif type_metric == 'AirwayVolumeLeakage':
            return AirwayVolumeLeakage_train(is_mask_exclude=is_mask_exclude)
        elif type_metric == 'AirwayCentrelineLeakage':
            return AirwayCentrelineLeakage_train(is_mask_exclude=is_mask_exclude)
        else:
            message = 'Choice Metric for Training not found. Metrics available: %s' % (', '.join(LIST_AVAIL_METRICS_TRAIN))
            catch_error_exception(message)


def get_network(type_network: str,
                size_image_in: Tuple[int, int, int],
                num_levels: int = 5,
                num_featmaps_in: int = 16,
                num_channels_in: int = 1,
                num_classes_out: int = 1,
                is_use_valid_convols: bool = False,
                **kwargs) -> ConvNetBase:
    if type_network == 'UNet3D_Original':
        return UNet3D_Original(size_image_in,
                               num_featmaps_in=num_featmaps_in,
                               num_channels_in=num_channels_in,
                               num_classes_out=num_classes_out)

    elif type_network == 'UNet3D_General':
        type_activate_hidden = kwargs['type_activate_hidden'] if 'type_activate_hidden' in kwargs.keys() else 'relu'
        type_activate_output = kwargs['type_activate_output'] if 'type_activate_output' in kwargs.keys() else 'sigmoid'
        num_featmaps_levels = kwargs['num_featmaps_levels'] if 'num_featmaps_levels' in kwargs.keys() else None
        is_use_dropout = kwargs['is_use_dropout'] if 'is_use_dropout' in kwargs.keys() else False
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.2
        is_use_batchnormalize = kwargs['is_use_batchnormalize'] if 'is_use_batchnormalize' in kwargs.keys() else False

        return UNet3D_General(size_image_in,
                              num_levels,
                              num_featmaps_in=num_featmaps_in,
                              num_channels_in=num_channels_in,
                              num_classes_out=num_classes_out,
                              is_use_valid_convols=is_use_valid_convols,
                              type_activate_hidden=type_activate_hidden,
                              type_activate_output=type_activate_output,
                              num_featmaps_levels=num_featmaps_levels,
                              is_use_dropout=is_use_dropout,
                              dropout_rate=dropout_rate,
                              is_use_batchnormalize=is_use_batchnormalize)

    elif type_network == 'UNet3D_Plugin':
        return UNet3D_Plugin(size_image_in,
                             num_levels,
                             num_featmaps_in=num_featmaps_in,
                             num_channels_in=num_channels_in,
                             num_classes_out=num_classes_out,
                             is_use_valid_convols=is_use_valid_convols)
    else:
        message = 'Choice Network not found. Networks available: %s' % (', '.join(LIST_AVAIL_NETWORKS))
        catch_error_exception(message)


def get_optimizer(type_optimizer: str, learn_rate: float, **kwargs):
    if type_optimizer == 'SGD':
        return SGD(learn_rate, **kwargs)
    elif type_optimizer == 'SGD_mom':
        return SGD_mom(learn_rate, **kwargs)
    elif type_optimizer == 'Adagrad':
        return Adagrad(learn_rate, **kwargs)
    elif type_optimizer == 'RMSprop':
        return RMSprop(learn_rate, **kwargs)
    elif type_optimizer == 'Adadelta':
        return Adadelta(learn_rate, **kwargs)
    elif type_optimizer == 'Adam':
        return Adam(learn_rate, **kwargs)
    else:
        message = 'Choice Optimizer not found. Optimizers available: %s' % (', '.join(LIST_AVAIL_OPTIMIZERS))
        catch_error_exception(message)


if TYPE_DNNLIB_USED == 'Pytorch':
    from models.pytorch.modeltrainer import ModelTrainer
elif TYPE_DNNLIB_USED == 'Keras':
    from models.keras.modeltrainer import ModelTrainer