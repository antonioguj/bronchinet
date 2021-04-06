
from typing import List

from models.callbacks import RecordLossHistoryBase, ModelCheckpointBase
from models.metrics import MetricBase


class RecordLossHistory(RecordLossHistoryBase):

    def __init__(self,
                 loss_filename: str,
                 list_metrics: List[MetricBase] = None,
                 is_hist_validation: bool = True
                 ) -> None:
        super(RecordLossHistory, self).__init__(loss_filename, list_metrics,
                                                is_hist_validation=is_hist_validation)

    def on_train_begin(self, *args, **kwargs) -> None:
        super(RecordLossHistory, self).on_train_begin()

    def on_epoch_end(self, *args, **kwargs) -> None:
        epoch = args[0]
        data_output = args[1]
        super(RecordLossHistory, self).on_epoch_end(epoch, data_output)


class ModelCheckpoint(ModelCheckpointBase):

    def __init__(self,
                 model_filename: str,
                 model_trainer,
                 freq_save_model: int = 1,
                 type_save_model: str = 'full_model',
                 update_filename_epoch: bool = False
                 ) -> None:
        super(ModelCheckpoint, self).__init__(model_filename,
                                              model_trainer,
                                              freq_save_model=freq_save_model,
                                              type_save_model=type_save_model,
                                              update_filename_epoch=update_filename_epoch)

    def on_train_begin(self, *args, **kwargs) -> None:
        super(ModelCheckpoint, self).on_train_begin()

    def on_epoch_end(self, *args, **kwargs) -> None:
        epoch = args[0]
        super(ModelCheckpoint, self).on_epoch_end(epoch)
