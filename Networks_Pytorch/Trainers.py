#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.Constants import *
from Common.FunctionsUtil import *
from Networks_Pytorch.Metrics import *
from Networks_Pytorch.Networks import *
from Networks_Pytorch.Optimizers import *
import torch.nn as nn
import torch
from torchsummary import summary
from datetime import datetime as dt
from tqdm import tqdm
from collections import OrderedDict


class Trainer(object):

    def __init__(self, model_net,
                 optimizer,
                 loss_fun,
                 metrics= None,
                 callbacks= None):
        self.model_net = model_net
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self.metrics = metrics
        self.callbacks = callbacks

        self.device = self.get_device()
        self.model_net = self.model_net.to(self.device)


    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _criterion(self, prediction, ground_truth):
        return self.loss_fun.forward(ground_truth, prediction)


    def _train_epoch(self):
        if self.max_steps_epoch and self.max_steps_epoch<len(self.train_data_generator):
            num_batches = self.max_steps_epoch
        else:
            num_batches = len(self.train_data_generator)

        progressbar = tqdm(total= num_batches,
                           desc= 'Epochs {}/{}'.format(self.epoch_count, self.num_epochs),
                           bar_format= '{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]')

        # time_loaddata = 0.0
        # time_train = 0.0
        sumrun_loss = 0.0
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
            pred_batch = self.model_net(x_batch)
            loss = self._criterion(pred_batch, y_batch)
            loss.backward()
            self.optimizer.step()
            # loss.detach()
            sumrun_loss += loss.item()
            # time_now = dt.now()
            # time_train += (time_now - time_train_ini).seconds

            loss_partial = sumrun_loss/(icount_batch+1)
            progressbar.set_postfix(loss='{0:1.5f}'.format(loss_partial))
            progressbar.update(1)

            icount_batch += 1
            if icount_batch > num_batches:
                break
        # endfor

        total_loss = sumrun_loss/icount_batch
        return total_loss


    def _validation_epoch(self):
        if self.max_steps_epoch and self.max_steps_epoch < len(self.valid_data_generator):
            num_batches = self.max_steps_epoch
        else:
            num_batches = len(self.valid_data_generator)

        progressbar = tqdm(total= num_batches, desc= 'Validation', leave= False)

        # time_loaddata = 0.0
        # time_valid = 0.0
        sumrun_loss = 0.0
        # time_loaddata_ini = dt.now()
        # time_train_ini = dt.now()
        # run a validation pass on the current epoch
        icount_batch = 0
        for (x_batch, y_batch) in self.valid_data_generator:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            # time_now = dt.now()
            # time_loaddata += (time_now - time_loaddata_ini).seconds

            pred_batch = self.model_net(x_batch)
            loss = self._criterion(pred_batch, y_batch)
            loss.backward()
            # loss.detach()
            sumrun_loss += loss.item()
            # time_now = dt.now()
            # time_valid += (time_now - time_train_ini).seconds

            progressbar.update(1)

            icount_batch += 1
            if icount_batch > num_batches:
                break
        # endfor

        total_loss = sumrun_loss/icount_batch
        return total_loss


    def _run_epoch(self):
        # switch to train mode
        self.model_net = self.model_net.train()

        # run a train pass on the current epoch
        self.train_loss = self._train_epoch()

        if self.valid_data_generator:
            # switch to evaluate mode
            self.model_net = self.model_net.eval()

            # run the validation pass
            self.valid_loss = self._validation_epoch()

        self.output_loss_end_epoch()

        # run callbacks
        if self.callbacks:
           self._run_callbacks()
        #endfor


    def _run_prediction(self):
        num_batches = len(self.test_data_generator)
        size_output_batch = self.model_net.get_size_output()

        out_prediction = np.ndarray([num_batches] + size_output_batch, dtype=FORMATPROBABILITYDATA)

        progressbar = tqdm(total= num_batches, desc='Prediction')

        # run prediction pass
        for i_batch, (x_batch, y_batch) in enumerate(self.test_data_generator):
            x_batch = x_batch.to(self.device)
            # time_now = dt.now()
            # time_loaddata += (time_now - time_loaddata_ini).seconds

            pred_batch = self.model_net(x_batch)
            out_prediction[i_batch] = pred_batch.detach().to('cpu')
            # time_now = dt.now()
            # time_valid += (time_now - time_train_ini).seconds

            progressbar.update(1)
        #endfor

        # rollaxis to output in "channels_last"
        ndim_out = len(out_prediction.shape)
        return np.rollaxis(out_prediction, 1, ndim_out)


    def train(self, train_data_generator,
              num_epochs= 1,
              max_steps_epoch= None,
              valid_data_generator= None,
              initial_epoch= 0):
        self.num_epochs = num_epochs
        self.max_steps_epoch = max_steps_epoch
        self.epoch_count = initial_epoch
        self.train_data_generator = train_data_generator
        self.valid_data_generator = valid_data_generator

        # run training algorithm
        for i_epoch in range(initial_epoch, num_epochs):
            self._run_epoch()
            self.epoch_count += 1

            # write loss history
            self.update_losshistory_file()
            # save model
            self.save_model_full()
        #endfor

    def predict(self, test_data_generator):
        self.test_data_generator = test_data_generator

        # switch to evaluate mode
        self.model_net = self.model_net.eval()

        return self._run_prediction()


    def _run_callbacks(self):
        for callback in self.callbacks:
            callback()
        #endfor


    def output_loss_end_epoch(self):
        print
        print("train loss = {:03f}".format(self.train_loss))
        if self.valid_data_generator:
            print("valid loss = {:03f}".format(self.valid_loss))

    def setup_losshistory_filepath(self, filepath, relfilename= 'lossHistory.txt'):
        self.losshistory_filename = joinpathnames(filepath, relfilename)
        strheader = '/epoch/ /loss/ /val_loss/\n'
        self.fout = open(self.losshistory_filename, 'w')
        self.fout.write(strheader)
        self.fout.close()

    def update_losshistory_file(self):
        strdataline = '%s %s %s\n' %(self.epoch_count,
                                     self.train_loss,
                                     self.valid_loss)
        self.fout = open(self.losshistory_filename, 'a')
        self.fout.write(strdataline)
        self.fout.close()


    def setup_savemodel_filepath(self, filepath, relfilename= 'model_%0.2i_%0.5f_%0.5f.pt'):
        self.modelsave_filename = joinpathnames(filepath, relfilename)

    def save_model_only_weights(self):
        filename = self.modelsave_filename %(self.epoch_count, self.train_loss, self.valid_loss)
        torch.save(self.model_net.state_dict(), filename)

    def save_model_full(self):
        filename = self.modelsave_filename % (self.epoch_count, self.train_loss, self.valid_loss)
        torch.save({'model_desc': self.model_net.get_arch_desc(),
                    'model_state_dict': self.model_net.state_dict(),
                    'optimizer_desc': 'Adam',
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss_fun_desc': [self.loss_fun.__class__.__name__, {'is_masks_exclude': self.loss_fun.is_masks_exclude}]},
                   filename)

    def load_model_only_weights(self, filename):
        self.model_net.load_state_dict(torch.load(filename, map_location= self.device))

    @staticmethod
    def load_model_full(filename):
        trainer_desc = torch.load(filename, map_location= Trainer.get_device())
        # create new model
        model_type = trainer_desc['model_desc'][0]
        model_input_args = trainer_desc['model_desc'][1]
        model_net = NeuralNetwork.get_create_model(model_type, model_input_args)
        model_net.load_state_dict(trainer_desc['model_state_dict'])
        # CHECK THIS OUT !!!
        model_net.cuda()

        # create new optimizer
        optimizer_type = trainer_desc['optimizer_desc']
        optimizer = DICTAVAILOPTIMIZERS(optimizer_type, model_params= model_net.parameters(), lr=0.0)
        optimizer.load_state_dict(trainer_desc['optimizer_state_dict'])

        # create nwe loss function
        loss_fun_type = trainer_desc['loss_fun_desc'][0]
        loss_fun_input_args = trainer_desc['loss_fun_desc'][1]
        loss_fun = DICTAVAILLOSSFUNS(loss_fun_type, is_masks_exclude= loss_fun_input_args['is_masks_exclude'])

        # create and return new Trainer
        return Trainer(model_net, optimizer, loss_fun)

    def get_summary_model(self):
        summary(self.model_net, tuple(self.model_net.get_size_input()))


