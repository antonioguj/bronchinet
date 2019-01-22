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
from datetime import datetime as dt
from tqdm import tqdm
from collections import OrderedDict


class TrainManager(object):

    def __init__(self, model_net,
                 optimizer,
                 loss_fun,
                 metrics,
                 callbacks= None):
        self.model_net = model_net
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self.metrics = metrics
        self.callbacks = callbacks

        self.device = self.get_device()
        self.model_net = self.model_net.to(self.device)

        # self._criterion = nn.BCELoss()

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _criterion(self, prediction, ground_truth):
        return self.loss_fun.forward(ground_truth, prediction)


    def _train_epoch(self):
        # time_loaddata = 0.0
        # time_train = 0.0
        sumrun_loss = 0.0
        progressbar = tqdm(total= self.num_batches_train,
                           desc= 'Epochs {}/{}'.format(self.epoch_count, self.num_epochs),
                           bar_format= '{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]')

        # time_loaddata_ini = dt.now()
        # time_train_ini = dt.now()

        # run a train pass on the current epoch
        icount_batch = 0
        for (x_batch, y_batch) in self.train_data_generator:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            # time_now = dt.now()
            # time_loaddata += (time_now - time_loaddata_ini).seconds

            self.optimizer.zero_grad()
            output = self.model_net(x_batch)
            loss = self._criterion(output, y_batch)
            loss.backward()
            self.optimizer.step()
            # loss.detach()
            sumrun_loss += loss.item()
            # time_now = dt.now()
            # time_train += (time_now - time_train_ini).seconds

            icount_batch += 1
            if icount_batch > self.num_batches_train:
                break

            loss_partial = sumrun_loss / icount_batch
            progressbar.set_postfix(loss='{0:1.5f}'.format(loss_partial))
            progressbar.update(1)
        # endfor

        total_loss = sumrun_loss / icount_batch
        return total_loss


    def _validation_epoch(self):
        # time_loaddata = 0.0
        # time_valid = 0.0
        sumrun_loss = 0.0
        progressbar = tqdm(total= self.num_batches_valid, desc= 'Validation', leave= False)

        # time_loaddata_ini = dt.now()
        # time_train_ini = dt.now()

        # run a validation pass on the current epoch
        icount_batch = 0
        for (x_batch, y_batch) in self.valid_data_generator:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            # time_now = dt.now()
            # time_loaddata += (time_now - time_loaddata_ini).seconds

            output = self.model_net(x_batch)
            loss = self._criterion(output, y_batch)
            loss.backward()
            # loss.detach()
            sumrun_loss += loss.item()
            # time_now = dt.now()
            # time_valid += (time_now - time_train_ini).seconds

            icount_batch += 1
            if icount_batch > self.num_batches_valid:
                break

            progressbar.update(1)
        # endfor

        total_loss = sumrun_loss / icount_batch
        return total_loss


    def _calc_prediction(self):
        progressbar = tqdm(total= self.num_batches_test, desc='Prediction')

        # run prediction pass
        for (x_batch, y_batch) in self.test_data_generator:
            x_batch = x_batch.to(self.device)

            output = self.model_net(x_batch)


    def _run_callbacks(self, train_loss, valid_loss):
        for callback in self.callbacks:
            callback(model_net=self.model_net,
                     epoch_count=self.epoch_count,
                     train_loss=train_loss,
                     valid_loss=valid_loss)
        #endfor


    def _run_epoch(self):
        # switch to train mode
        self.model_net = self.model_net.train()

        # run a train pass on the current epoch
        train_loss = self._train_epoch()

        if self.valid_data_generator:
            # switch to evaluate mode
            self.model_net = self.model_net.eval()

            # run the validation pass
            valid_loss = self._validation_epoch()

        print("train loss = {:03f}".format(train_loss))
        if self.valid_data_generator:
            print("valid loss = {:03f}".format(valid_loss))

        # callbacks
        if self.callbacks:
           self._run_callbacks(train_loss, valid_loss)
        #endfor


    def train(self, train_data_generator,
              num_epochs= 1,
              max_steps_epoch= None,
              valid_data_generator= None,
              initial_epoch= 0):
        self.num_epochs = num_epochs
        self.epoch_count = initial_epoch
        self.train_data_generator = train_data_generator
        self.valid_data_generator = valid_data_generator

        if max_steps_epoch and max_steps_epoch<len(self.train_data_generator):
            self.num_batches_train = max_steps_epoch
        else:
            self.num_batches_train = len(self.train_data_generator)

        if self.valid_data_generator:
            if max_steps_epoch and max_steps_epoch<len(self.valid_data_generator):
                self.num_batches_valid = max_steps_epoch
            else:
                self.num_batches_valid = len(self.valid_data_generator)
        else:
            self.num_batches_valid = 0


        # run training algorithm
        for i_epoch in range(initial_epoch, num_epochs):
            self._run_epoch()
            self.epoch_count += 1
        #endfor


    def predict(self, test_data_generator):
        self.test_data_generator = test_data_generator
        self.num_batches_test = len(self.test_data_generator)

        # switch to evaluate mode
        self.model_net = self.model_net.eval()

        return self._calc_prediction()
