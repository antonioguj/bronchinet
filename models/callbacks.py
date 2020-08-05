
from typing import List

from common.constant import TYPE_DNNLIB_USED
from models.metrics import MetricBase
if TYPE_DNNLIB_USED == 'Pytorch':
    from models.pytorch.callbacks import RecordLossHistory
elif TYPE_DNNLIB_USED == 'Keras':
    from models.keras.callbacks import RecordLossHistory, EarlyStopping, ModelCheckpoint

class CallbackBase(object):

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def on_train_begin(self, *args, **kwargs):
        raise NotImplementedError

    def on_epoch_end(self, *args, **kwargs):
        raise NotImplementedError


class RecordLossHistoryBase(CallbackBase):

    def __init__(self,
                 loss_filename: str,
                 list_metrics: List[MetricBase] = None
                 ) -> None:
        self._loss_filename = loss_filename
        self._names_metrics_funs = []
        if list_metrics:
            for imetric in list_metrics:
                new_names_metrics_funs = ['%s' % (imetric._name_fun_out), 'val_%s' % (imetric._name_fun_out)]
                self._names_metrics_funs += new_names_metrics_funs

    def on_train_begin(self) -> None:
        list_header = ['/epoch/', '/loss/', '/val_loss/']
        if self._names_metrics_funs:
            list_header += ['/%s/' % (iname) for iname in self._names_metrics_funs]
        str_header = ' '.join(list_header) + '\n'

        fout = open(self._loss_filename, 'w')
        fout.write(str_header)
        fout.close()

    def on_epoch_end(self, epoch: int, data_output: List[float]) -> None:
        list_data_line = ['%0.3d' % (epoch)] + ['%0.6f' % (elem) for elem in data_output]
        str_data_line = ' '.join(list_data_line) + '\n'

        fout = open(self._loss_filename, 'a')
        fout.write(str_data_line)
        fout.close()


class EarlyStoppingBase(CallbackBase):

    def __init__(self,
                 delta: float = 0.005,
                 patience: int = 10
                 ) -> None:
        self._threshold = (1.0 - delta)
        self._patience = patience

    def on_train_begin(self) -> None:
        self._best_epoch = 0
        self._best_valid_loss = 1.0e+03
        self._waiting = -1.0e+03

    def on_epoch_end(self, epoch: int, valid_loss) -> None:
        if (valid_loss < self._threshold * self._best_valid_loss):
            self._best_epoch = epoch
            self._best_valid_loss = valid_loss
            self._waiting = 0
        else:
            self._waiting += 1