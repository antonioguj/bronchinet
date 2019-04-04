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
        
        #self.device = self.get_device()
        #self.model_net = self.model_net.to(self.device)
        self.model_net.cuda()


    @staticmethod
    def get_device():
        return NotImplemented
        #return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        time_compute = 0.0
        time_total_ini = dt.now()
        sumrun_loss = 0.0

        # run a train pass on the current epoch
        i_batch = 0
        for (x_batch, y_batch) in self.train_data_generator:
            #x_batch = x_batch.to(self.device)
            #y_batch = y_batch.to(self.device)
            x_batch.cuda()
            y_batch.cuda()

            time_ini = dt.now()
            self.optimizer.zero_grad()
            pred_batch = self.model_net(x_batch)
            loss = self._criterion(pred_batch, y_batch)
            loss.backward()
            self.optimizer.step()
            # loss.detach()
            sumrun_loss += loss.item()
            time_now = dt.now()
            time_compute += (time_now - time_ini).seconds

            loss_partial = sumrun_loss/(i_batch+1)
            progressbar.set_postfix(loss='{0:1.5f}'.format(loss_partial))
            progressbar.update(1)

            i_batch += 1
            if i_batch > num_batches:
                break
        # endfor

        time_now = dt.now()
        time_total = (time_now - time_total_ini).seconds
        time_loaddata = time_total - time_compute

        print("\ntime total = {0:.3f}".format(time_total))
        print("time loaddata / compute = {0:.3f} / {1:.3f}".format(time_loaddata, time_compute))

        total_loss = sumrun_loss/num_batches
        return total_loss


    def _validation_epoch(self):
        if self.max_steps_epoch and self.max_steps_epoch < len(self.valid_data_generator):
            num_batches = self.max_steps_epoch
        else:
            num_batches = len(self.valid_data_generator)

        progressbar = tqdm(total= num_batches, desc= 'Validation', leave= False)

        time_compute = 0.0
        time_total_ini = dt.now()
        sumrun_loss = 0.0

        # run a validation pass on the current epoch
        i_batch = 0
        for (x_batch, y_batch) in self.valid_data_generator:
            #x_batch = x_batch.to(self.device)
            #y_batch = y_batch.to(self.device)
            x_batch.cuda()
            y_batch.cuda()

            time_ini = dt.now()
            pred_batch = self.model_net(x_batch)
            loss = self._criterion(pred_batch, y_batch)
            loss.backward()
            # loss.detach()
            sumrun_loss += loss.item()
            time_now = dt.now()
            time_compute += (time_now - time_ini).seconds

            progressbar.update(1)

            i_batch += 1
            if i_batch > num_batches:
                break
        # endfor

        time_now = dt.now()
        time_total = (time_now - time_total_ini).seconds
        time_loaddata = time_total - time_compute

        print("\ntime total = {0:.3f}".format(time_total))
        print("time loaddata / compute = {0:.3f} / {1:.3f}".format(time_loaddata, time_compute))

        total_loss = sumrun_loss/num_batches
        return total_loss


    def _run_epoch(self):
        # switch to train mode
        self.model_net = self.model_net.train()

        # run a train pass on the current epoch
        self.train_loss = self._train_epoch()

        if self.valid_data_generator and \
            (self.epoch_count % self.freq_validate_model == 0):
            # switch to evaluate mode
            self.model_net = self.model_net.eval()

            # run the validation pass
            self.valid_loss = self._validation_epoch()

        # run callbacks
        if self.callbacks:
           self._run_callbacks()
        #endfor


    def _run_prediction(self):
        num_batches = len(self.test_data_generator)
        size_output_batch = self.model_net.get_size_output()

        out_prediction = np.ndarray([num_batches] + size_output_batch, dtype=FORMATPROBABILITYDATA)

        progressbar = tqdm(total= num_batches, desc='Prediction')

        time_compute = 0.0
        time_total_ini = dt.now()

        # run prediction pass
        for i_batch, (x_batch, y_batch) in enumerate(self.test_data_generator):
            #x_batch = x_batch.to(self.device)
            x_batch.cuda()

            time_ini = dt.now()
            pred_batch = self.model_net(x_batch)
            #out_prediction[i_batch] = pred_batch.detach().to('cpu')
            out_prediction[i_batch] = pred_batch.detach().cpu()
            time_now = dt.now()
            time_compute_i = (time_now - time_ini).seconds
            time_compute += time_compute_i

            progressbar.update(1)
        #endfor

        time_now = dt.now()
        time_total = (time_now - time_total_ini).seconds
        time_loaddata = time_total - time_compute

        print("\ntime total = {0:.3f}".format(time_total))
        print("time loaddata / compute = {0:.3f} / {1:.3f}".format(time_loaddata, time_compute))

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

        self.train_loss = 0.0
        self.valid_loss = 0.0

        # run training algorithm
        for i_epoch in range(initial_epoch, num_epochs):
            self._run_epoch()
            self.epoch_count += 1

            # write loss history
            print("\ntrain loss = {0:.3f}".format(self.train_loss))
            if self.valid_data_generator:
                print("valid loss = {0:.3f}".format(self.valid_loss))

            if self.is_write_lossfile:
                self.update_losshistory_file()

            # save model
            if self.is_save_model:
                self.save_models()
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


    def setup_losshistory_filepath(self, filepath,
                                   relfilename= 'lossHistory.txt',
                                   isexists_lossfile= False):
        self.is_write_lossfile = True
        self.losshistory_filename = joinpathnames(filepath, relfilename)
        if isexists_lossfile:
            self.fout = open(self.losshistory_filename, 'a')
        else:
            self.fout = open(self.losshistory_filename, 'w')
            strheader = '/epoch/ /loss/ /val_loss/\n'
            self.fout.write(strheader)
        self.fout.close()

    def update_losshistory_file(self):
        self.fout = open(self.losshistory_filename, 'a')
        strdataline = '%s %s %s\n' %(self.epoch_count,
                                     self.train_loss,
                                     self.valid_loss)
        self.fout.write(strdataline)
        self.fout.close()


    def setup_validate_model(self, freq_validate_model= 1):
        self.freq_validate_model = freq_validate_model

    def setup_savemodel_filepath(self, filepath,
                                 type_save_models= 'full_model',
                                 freq_save_intermodels= None):
        self.is_save_model = True
        self.filepath = filepath
        self.type_save_models = type_save_models
        self.freq_save_intermodels = freq_save_intermodels

    def save_models(self):
        relfilename = 'model_last.pt'
        model_filename = joinpathnames(self.filepath, relfilename)
        if self.type_save_models == 'only_weights':
            self.save_model_only_weights(model_filename)
        elif self.type_save_models == 'full_model':
            self.save_model_full(model_filename)

        if self.freq_save_intermodels:
            if (self.epoch_count%self.freq_save_intermodels==0):
                relfilename = 'model_e%0.2i.pt' %(self.epoch_count)
                model_filename = joinpathnames(self.filepath, relfilename)
                if self.type_save_models == 'only_weights':
                    self.save_model_only_weights(model_filename)
                elif self.type_save_models == 'full_model':
                    self.save_model_full(model_filename)

    def save_model_only_weights(self, filename):
        torch.save(self.model_net.state_dict(), filename)

    def save_model_full(self, filename):
        torch.save({'model_desc': self.model_net.get_arch_desc(),
                    'model_state_dict': self.model_net.state_dict(),
                    'optimizer_desc': 'Adam',
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss_fun_desc': [self.loss_fun.__class__.__name__, {'is_masks_exclude': self.loss_fun.is_masks_exclude}]},
                   filename)


    def load_model_only_weights(self, filename):
        self.model_net.load_state_dict(torch.load(filename, map_location= 'cuda:0'))#map_location= self.device))

    @staticmethod
    def load_model_full(filename):
        trainer_desc = torch.load(filename, map_location= 'cuda:0')#map_location= Trainer.get_device())
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


