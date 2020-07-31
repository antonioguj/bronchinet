
from typing import List, Any, Callable
from tensorflow.keras import callbacks as callbacks_Keras

from common.function_util import join_path_names, flatten_listoflists
from networks.callbacks import Callback


class RecordLossHistory(Callback, callbacks_Keras.Callback):

    def __init__(self,
                 filepath: str,
                 relfilename: str,
                 metrics_funs: List[Callable] = None
                 ) -> None:
        self._filename = join_path_names(filepath, relfilename)
        if metrics_funs:
            self._names_metrics_funs = list(map(lambda fun: ['%s' % (fun.__name__), 'val_%s' % (fun.__name__)], metrics_funs))
            self._names_metrics_funs = flatten_listoflists(self._names_metrics_funs)
        else:
            self._names_metrics_funs = []

    def on_train_begin(self, logs = None) -> None:
        str_header = '/epoch/ /loss/ /val_loss/'
        if self._names_metrics_funs:
            str_header += ' ' + ' '.join(['/%s/' % (fun) for fun in self._names_metrics_funs])
        str_header += '\n'

        fout = open(self._filename, 'w')
        fout.write(str_header)
        fout.close()

    def on_epoch_end(self, epoch: int, logs = None) -> None:
        str_data_line = '%s %s %s' %(epoch, logs.get('loss'), logs.get('val_loss'))
        if self._names_metrics_funs:
            str_data_line += ' ' + ' '.join(['%s' % (logs.get(fun)) for fun in self._names_metrics_funs])
        str_data_line += '\n'

        fout = open(self._filename, 'a')
        fout.write(str_data_line)
        fout.close()


class EarlyStopping(Callback, callbacks_Keras.Callback):

    def __init__(self,
                 delta: float = 0.005,
                 patience: int = 10
                 ) -> None:
        self._threshold = (1.0 - delta)
        self._patience = patience

    def on_train_begin(self, logs = None) -> None:
        self._best_epoch = 0
        self._best_val_loss = 1.0e+03
        self._waiting = -1.0e+03

    def on_epoch_end(self, epoch: int, logs = None) -> None:
        this_val_loss = logs.get('val_loss')
        if( this_val_loss < self._threshold * self._best_val_loss):
            self._best_epoch = epoch
            self._best_val_loss = this_val_loss
            self._waiting = 0
        else:
            self._waiting += 1
            if(self._waiting > self._patience):
                print("Early stopping training. Save model fom epoch \'%s\' and validation loss \'%s\'..."
                      % (self._best_epoch, self._best_val_loss))
                self.model.stop_training = True