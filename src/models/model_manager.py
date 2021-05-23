
from typing import Tuple, Union, Any

from common.constant import TYPE_DNNLIB_USED
from common.exceptionmanager import catch_error_exception
if TYPE_DNNLIB_USED == 'Pytorch':
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
        LIST_AVAIL_METRICS as LIST_AVAIL_METRICS_TRAIN
    from models.pytorch.networks import UNet, UNet3DOriginal, UNet3DGeneral, UNet3DPlugin, LIST_AVAIL_NETWORKS
    from models.pytorch.optimizers import get_sgd, get_sgdmom, get_rmsprop, get_adagrad, get_adadelta, get_adam, \
        LIST_AVAIL_OPTIMIZERS
    from models.pytorch.networkchecker import NetworkChecker
elif TYPE_DNNLIB_USED == 'Keras':
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
        LIST_AVAIL_METRICS as LIST_AVAIL_METRICS_TRAIN
    from models.keras.networks import UNet, UNet3DOriginal, UNet3DGeneral, UNet3DPlugin, LIST_AVAIL_NETWORKS
    from models.keras.optimizers import get_sgd, get_sgdmom, get_rmsprop, get_adagrad, get_adadelta, get_adam, \
        LIST_AVAIL_OPTIMIZERS
    from models.keras.networkchecker import NetworkChecker
from models.metrics import MetricBase, MeanSquaredError, MeanSquaredErrorLogarithmic, \
    BinaryCrossEntropy, WeightedBinaryCrossEntropy, WeightedBinaryCrossEntropyFixedWeights, \
    DiceCoefficient, TruePositiveRate, TrueNegativeRate, FalsePositiveRate, FalseNegativeRate, \
    AirwayMetricBase, AirwayCompleteness, AirwayVolumeLeakage, AirwayCentrelineLeakage, AirwayTreeLength, \
    AirwayCentrelineDistanceFalseNegativeError, AirwayCentrelineDistanceFalsePositiveError, \
    LIST_AVAIL_METRICS
from models.networks import ConvNetBase


def get_metric(type_metric: str,
               is_mask_exclude: bool = False,
               **kwargs) -> Union[MetricBase, AirwayMetricBase]:
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
    elif type_metric == 'AirwayTreeLength':
        return AirwayTreeLength(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'AirwayCentrelineDistanceFalsePositiveError':
        return AirwayCentrelineDistanceFalsePositiveError(is_mask_exclude=is_mask_exclude)
    elif type_metric == 'AirwayCentrelineDistanceFalseNegativeError':
        return AirwayCentrelineDistanceFalseNegativeError(is_mask_exclude=is_mask_exclude)
    else:
        message = 'Choice Metric not found: \'%s\'. Metrics available: \'%s\'' \
                  % (type_metric, ', '.join(LIST_AVAIL_METRICS))
        catch_error_exception(message)


def get_metric_train(type_metric: str,
                     is_mask_exclude: bool = False,
                     **kwargs) -> Metric_train:
    if 'Combined_' in type_metric:
        splitels_type_metric = type_metric.split('_')
        if len(splitels_type_metric) != 3:
            message = 'For combined Loss, set metric name as \'Combi_<name_metric1>_<name_metric2>\'. ' \
                      'Wrong name now: \'%s\'' % (type_metric)
            catch_error_exception(message)
        type_metric_1 = splitels_type_metric[1]
        type_metric_2 = splitels_type_metric[2]
        weight_combined_loss = kwargs['weight_combined_loss']
        print("Chosen combined Loss with metrics \'%s\' and \'%s\', and weighting between 2nd and 1st metric: \'%s\'..."
              % (type_metric_1, type_metric_2, weight_combined_loss))

        metrics_1 = get_metric_train(type_metric_1, is_mask_exclude)
        metrics_2 = get_metric_train(type_metric_2, is_mask_exclude)
        return CombineTwoMetrics_train(metrics_1, metrics_2, weight_metric2over1=weight_combined_loss)
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
        else:
            message = 'Choice Metric for Training not found: \'%s\'. Metrics available: \'%s\'' \
                      % (type_metric, ', '.join(LIST_AVAIL_METRICS_TRAIN))
            catch_error_exception(message)


def get_network(type_network: str,
                size_image_in: Union[Tuple[int, int, int], Tuple[int, int]],
                num_featmaps_in: int = 16,
                num_channels_in: int = 1,
                num_classes_out: int = 1,
                is_use_valid_convols: bool = False,
                **kwargs) -> ConvNetBase:
    if type_network == 'UNet3DOriginal':
        return UNet3DOriginal(size_image_in,
                              num_featmaps_in=num_featmaps_in,
                              num_channels_in=num_channels_in,
                              num_classes_out=num_classes_out)

    elif type_network == 'UNet3DGeneral':
        num_levels = \
            kwargs['num_levels'] if 'num_levels' in kwargs.keys() else UNet3DGeneral._num_levels_default
        type_activate_hidden = \
            kwargs['type_activate_hidden'] if 'type_activate_hidden' in kwargs.keys() \
            else UNet3DGeneral._type_activate_hidden_default
        type_activate_output = \
            kwargs['type_activate_output'] if 'type_activate_output' in kwargs.keys() \
            else UNet3DGeneral._type_activate_output_default
        is_use_dropout = \
            kwargs['is_use_dropout'] if 'is_use_dropout' in kwargs.keys() else False
        dropout_rate = \
            kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else UNet3DGeneral._dropout_rate_default
        is_use_batchnormalize = \
            kwargs['is_use_batchnormalize'] if 'is_use_batchnormalize' in kwargs.keys() else False

        return UNet3DGeneral(size_image_in,
                             num_levels=num_levels,
                             num_featmaps_in=num_featmaps_in,
                             num_channels_in=num_channels_in,
                             num_classes_out=num_classes_out,
                             is_use_valid_convols=is_use_valid_convols,
                             type_activate_hidden=type_activate_hidden,
                             type_activate_output=type_activate_output,
                             is_use_dropout=is_use_dropout,
                             dropout_rate=dropout_rate,
                             is_use_batchnormalize=is_use_batchnormalize)

    elif type_network == 'UNet3DPlugin':
        return UNet3DPlugin(size_image_in,
                            num_featmaps_in=num_featmaps_in,
                            num_channels_in=num_channels_in,
                            num_classes_out=num_classes_out,
                            is_use_valid_convols=is_use_valid_convols)
    else:
        message = 'Choice Network not found: \'%s\'. Networks available: \'%s\'' \
                  % (type_network, ', '.join(LIST_AVAIL_NETWORKS))
        catch_error_exception(message)


def get_optimizer(type_optimizer: str, learn_rate: float, **kwargs) -> Any:
    if type_optimizer == 'SGD':
        return get_sgd(learn_rate, **kwargs)
    elif type_optimizer == 'SGDmom':
        return get_sgdmom(learn_rate, **kwargs)
    elif type_optimizer == 'Adagrad':
        return get_adagrad(learn_rate, **kwargs)
    elif type_optimizer == 'RMSprop':
        return get_rmsprop(learn_rate, **kwargs)
    elif type_optimizer == 'Adadelta':
        return get_adadelta(learn_rate, **kwargs)
    elif type_optimizer == 'Adam':
        return get_adam(learn_rate, **kwargs)
    else:
        message = 'Choice Optimizer not found: \'%s\'. Optimizers available: \'%s\'' \
                  % (type_optimizer, ', '.join(LIST_AVAIL_OPTIMIZERS))
        catch_error_exception(message)


def get_network_checker(size_image_in: Union[Tuple[int, int, int], Tuple[int, int]],
                        in_network: UNet
                        ) -> NetworkChecker:
    return NetworkChecker(size_image_in, in_network)


if TYPE_DNNLIB_USED == 'Pytorch':
    from models.pytorch.modeltrainer import ModelTrainer
elif TYPE_DNNLIB_USED == 'Keras':
    from models.keras.modeltrainer import ModelTrainer


def get_model_trainer() -> ModelTrainer:
    return ModelTrainer()
