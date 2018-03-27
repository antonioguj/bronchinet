#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from keras import callbacks

# All derived from baseclass "callbacks.Callback"

class RecordLossHistory(callbacks.Callback):

    def __init__(self, filepath):
        self.filename = filepath + '/lossHistory.txt'

    def on_train_begin(self, logs=None):
        self.fout = open(self.filename, 'w')
        self.fout.write('/epoch/ /loss/ /val_loss/\n')
        self.fout.close()

    def on_epoch_end(self, epoch, logs=None):
        self.fout = open(self.filename, 'a')
        newdataline = '%s %s %s\n' %(epoch, logs.get('loss'), logs.get('val_loss'))
        self.fout.write( newdataline )
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
