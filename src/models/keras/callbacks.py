
from typing import List, Any
from tensorflow.keras import callbacks as callbacks_keras

from models.callbacks import RecordLossHistoryBase, EarlyStoppingBase, ModelCheckpointBase
from models.metrics import MetricBase


class RecordLossHistory(RecordLossHistoryBase, callbacks_keras.Callback):

    def __init__(self,
                 loss_filename: str,
                 list_metrics: List[MetricBase] = None,
                 is_hist_validation: bool = True,
                 is_restart_model: bool = False
                 ) -> None:
        super(RecordLossHistory, self).__init__(loss_filename, list_metrics,
                                                is_hist_validation=is_hist_validation)
        callbacks_keras.Callback.__init__(self)

        self._is_restart_model = is_restart_model

    def on_train_begin(self, logs: Any = None) -> None:
        if not self._is_restart_model:
            super(RecordLossHistory, self).on_train_begin()

    def on_epoch_end(self, epoch: int, logs: Any) -> None:
        data_output = [logs.get(iname) for iname in self._names_hist_fields]
        super(RecordLossHistory, self).on_epoch_end(epoch, data_output)


class EarlyStopping(EarlyStoppingBase, callbacks_keras.Callback):

    def __init__(self,
                 delta: float = 0.005,
                 patience: int = 10
                 ) -> None:
        super(EarlyStoppingBase, self).__init__(delta, patience)
        callbacks_keras.Callback.__init__(self)

    def on_train_begin(self, logs: Any = None) -> None:
        super(EarlyStoppingBase, self).on_train_begin()

    def on_epoch_end(self, epoch: int, logs: Any) -> None:
        valid_loss = logs.get('val_loss')
        super(EarlyStoppingBase, self).on_epoch_end(epoch, valid_loss)
        if (self._waiting > self._patience):
            print("Early stopping training. Save model fom epoch \'%s\' and validation loss \'%s\'..."
                  % (self._best_epoch, self._best_val_loss))
            self.model.stop_training = True


class ModelCheckpoint(ModelCheckpointBase, callbacks_keras.Callback):

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
        callbacks_keras.Callback.__init__(self)

    def on_train_begin(self, logs: Any = None) -> None:
        super(ModelCheckpoint, self).on_train_begin()

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        super(ModelCheckpoint, self).on_epoch_end(epoch)
