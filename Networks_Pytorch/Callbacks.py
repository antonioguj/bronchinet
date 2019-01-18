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
import torch


class Callback(object):
    def __call__(self, *args, **kwargs):
        raise NotImplemented


class RecordLossHistory(Callback):

    relfilename = 'lossHistory.txt'

    def __init__(self, filepath):
        self.filename = joinpathnames(filepath, self.relfilename)
        self.write_header()

    def write_header(self):
        strheader = '/epoch/ /loss/ /val_loss/\n'
        self.fout = open(self.filename, 'w')
        self.fout.write(strheader)
        self.fout.close()

    def __call__(self, *args, **kwargs):
        epoch_count = kwargs['epoch_count']
        train_loss = kwargs['train_loss']
        valid_loss = kwargs['valid_loss']
        strdataline = '%s %s %s\n' %(epoch_count, train_loss, valid_loss)
        self.fout = open(self.filename, 'a')
        self.fout.write(strdataline)
        self.fout.close()


class ModelCheckpoint(Callback):

    def __init__(self, filepath):
        self.filepath = filepath
        self.relfilename = 'model_%0.2i_%0.5f_%0.5f.hdf5'

    def __call__(self, *args, **kwargs):
        model_net = kwargs['model_net']
        epoch_count = kwargs['epoch_count']
        train_loss = kwargs['train_loss']
        valid_loss = kwargs['valid_loss']
        filename = joinpathnames(self.filepath, self.relfilename %(epoch_count, train_loss, valid_loss))
        torch.save(model_net.state_dict(), filename)