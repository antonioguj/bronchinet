#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.FunctionsUtil import *
from keras import callbacks


class RecordLossHistory(callbacks.Callback):

    relfilename = 'lossHistory.txt'

    def __init__(self, filepath, metrics_funs=None):
        self.filename = joinpathnames(filepath, self.relfilename)
        if metrics_funs:
            self.name_metrics_funs = list(map(lambda fun: ['%s'%(fun.__name__),'val_%s'%(fun.__name__)], metrics_funs))
            self.name_metrics_funs = flattenOutListOfLists(self.name_metrics_funs)
        else:
            self.name_metrics_funs = []

    def on_train_begin(self, logs=None):
        strheader = '/epoch/ /loss/ /val_loss/'
        if self.name_metrics_funs:
            strheader += ' ' + ' '.join(['/%s/' %(fun) for fun in self.name_metrics_funs])
        strheader += '\n'

        self.fout = open(self.filename, 'w')
        self.fout.write(strheader)
        self.fout.close()

    def on_epoch_end(self, epoch, logs=None):
        strdataline = '%s %s %s' %(epoch, logs.get('loss'), logs.get('val_loss'))
        if self.name_metrics_funs:
            strdataline += ' ' + ' '.join(['%s' %(logs.get(fun)) for fun in self.name_metrics_funs])
        strdataline += '\n'

        self.fout = open(self.filename, 'a')
        self.fout.write(strdataline)
        self.fout.close()


class EarlyStopping(callbacks.Callback):

    def __init__(self, delta=0.005, patience=10):
        self.threshold = (1.0-delta)
        self.patience  = patience

    def on_train_begin(self, logs=None):
        self.best_epoch = 0
        self.best_val_loss = 1.0e+03
        self.waiting = -1.0e+03

    def on_epoch_end(self, epoch, logs=None):
        this_val_loss = logs.get('val_loss')
        if( this_val_loss < self.threshold*self.best_val_loss ):
            self.best_epoch = epoch
            self.best_val_loss = this_val_loss
            self.waiting = 0
        else:
            self.waiting += 1
            if( self.waiting > self.patience ):
                print("Early stopping training. Save model fom epoch %s and validation loss %s"
                      %(self.best_epoch, self.best_val_loss))
                self.model.stop_training = True