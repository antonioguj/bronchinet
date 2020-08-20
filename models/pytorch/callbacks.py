
from typing import List

from models.callbacks import CallbackBase, RecordLossHistoryBase, EarlyStoppingBase
from models.metrics import MetricBase
#from models.modeltrainer import ModelTrainerBase


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


class ModelCheckpoint(CallbackBase):

    def __init__(self,
                 model_filename: str,
                 model_trainer,
                 freq_save_model: int = 1,
                 type_save_model: str = 'full_model',
                 update_filename_epoch: bool = False
                 ) -> None:
        self._model_filename = model_filename
        self._model_trainer = model_trainer
        self._freq_save_model = freq_save_model
        self._type_save_model = type_save_model
        self._update_filename_epoch = update_filename_epoch
        super(ModelCheckpoint, self).__init__()

    def on_train_begin(self, *args, **kwargs) -> None:
        pass

    def on_epoch_end(self, *args, **kwargs) -> None:
        epoch = args[0]
        if (epoch % self._freq_save_model == 0):
            if self._update_filename_epoch:
                model_filename_this = self._model_filename % (epoch)
            else:
                model_filename_this = self._model_filename

            if self._type_save_model == 'only_weights':
                self._model_trainer.save_model_only_weights(model_filename_this)
            elif self._type_save_model == 'full_model':
                self._model_trainer.save_model_full(model_filename_this)

