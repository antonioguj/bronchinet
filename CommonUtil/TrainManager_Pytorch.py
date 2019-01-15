#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.Constants import *
import torch.nn as nn
import torch


class TrainManager(object):

    def __init__(self, model_net, optimizer, loss_fun, metrics, callbacks=None):

        self.model_net = model_net
        self.optimizer = optimizer
        self.loss_fun  = loss_fun
        self.metrics   = metrics
        self.callbacks = callbacks

        self.device    = self.get_device()
        #self.model_net = self.model_net.to(self.device)


    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _criterion(self, prediction, ground_truth):
        return self.loss_fun.forward(prediction, ground_truth)

    def _run_callbacks(self):

        for callback in self.callbacks:
            callback()
        #endfor

    def _train_epoch(self):

        loss_total = 0.0
        # run a train pass on the current epoch
        for i, (x_batch, y_batch) in enumerate(self.train_data_generator):

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            output = self.model_net.forward(x_batch)
            loss   = self._criterion(output, y_batch)
            loss.backward()

            self.optimizer.zero_grad()
            self.optimizer.step()

            loss_total += loss.detach().item()
        # endfor

        return loss_total

    def _validation_epoch(self):

        loss_total = 0.0
        # run a train pass on the current epoch
        for i, (x_batch, y_batch) in enumerate(self.train_data_generator):

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            output = self.model_net.forward(x_batch)
            loss   = self._criterion(output, y_batch)
            loss.backward()

            loss_total += loss.detach().item()
        # endfor

        return loss_total

    def _run_epoch(self):

        # switch to train mode
        self.model_net.train()

        # run a train pass on the current epoch
        train_loss = self._train_epoch()

        # switch to evaluate mode
        self.model_net.eval()

        # run the validation pass
        valid_loss = self._validation_epoch()

        # callbacks
        self._run_callbacks()


    def train(self, train_data_generator, valid_data_generator, num_epochs, initial_epoch=0):

        self.train_data_generator = train_data_generator
        self.valid_data_generator = valid_data_generator

        for iepoch in range(initial_epoch, num_epochs):
            self._run_epoch()
        #endfor