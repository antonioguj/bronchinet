
from typing import List
from tensorflow.keras import callbacks as callbacks_keras

from models.callbacks import CallbackBase, RecordLossHistoryBase, EarlyStoppingBase
from models.metrics import MetricBase


class RecordLossHistory(RecordLossHistoryBase, callbacks_keras.Callback):

    def __init__(self,
                 loss_filename: str,
                 list_metrics: List[MetricBase] = None
                 ) -> None:
        super(RecordLossHistory, self).__init__(loss_filename, list_metrics)

    def on_train_begin(self, logs = None) -> None:
        super(RecordLossHistory, self).on_train_begin()

    def on_epoch_end(self, epoch: int, logs = None) -> None:
        data_output = [logs.get('loss'), logs.get('val_loss')] + [logs.get(iname) for iname in self._names_metrics_funs]
        super(RecordLossHistory, self).on_epoch_end(epoch, data_output)


class EarlyStopping(EarlyStoppingBase, callbacks_keras.Callback):

    def __init__(self,
                 delta: float = 0.005,
                 patience: int = 10
                 ) -> None:
        super(EarlyStoppingBase, self).__init__(delta, patience)

    def on_train_begin(self, logs = None) -> None:
        super(EarlyStoppingBase, self).on_train_begin()

    def on_epoch_end(self, epoch: int, logs = None) -> None:
        valid_loss = logs.get('val_loss')
        super(EarlyStoppingBase, self).on_epoch_end(epoch, valid_loss)
        if (self._waiting > self._patience):
            print("Early stopping training. Save model fom epoch \'%s\' and validation loss \'%s\'..."
                  % (self._best_epoch, self._best_val_loss))
            self.model.stop_training = True


class ModelCheckpoint(CallbackBase, callbacks_keras.ModelCheckpoint):

    def __init__(self, model_filename: str):
        super(ModelCheckpoint, self).__init__(model_filename, monitor='loss', verbose=0)