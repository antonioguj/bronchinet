
from typing import Union, Tuple, List
import numpy as np

import torch
from torchsummary import summary
from tqdm import tqdm

from common.constant import NAME_LOSSHISTORY_FILE, NAME_SAVEDMODEL_EPOCH_TORCH, NAME_SAVEDMODEL_LAST_TORCH
from common.exceptionmanager import catch_error_exception
from common.functionutil import join_path_names
from dataloaders.batchdatagenerator import BatchDataGenerator
from models.modeltrainer import ModelTrainerBase
from models.pytorch.callbacks import RecordLossHistory, ModelCheckpoint


class ModelTrainer(ModelTrainerBase):

    def __init__(self):
        super(ModelTrainer, self).__init__()
        #self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._device = 'cuda:0'

    def compile_model(self) -> None:
        pass

    def create_callbacks(self, models_path: str, **kwargs) -> None:
        self._list_callbacks = []

        losshist_filename = join_path_names(models_path, NAME_LOSSHISTORY_FILE)
        new_callback = RecordLossHistory(losshist_filename, self._list_metrics),
        self._list_callbacks.append(new_callback)

        freq_save_check_model = kwargs['freq_save_check_model'] if 'freq_save_check_model' in kwargs.keys() else 1
        freq_validate_model = kwargs['freq_validate_model'] if 'freq_validate_model' in kwargs.keys() else 1

        model_filename = join_path_names(models_path, NAME_SAVEDMODEL_EPOCH_TORCH)
        new_callback = ModelCheckpoint(model_filename, self,
                                       freq_save_model=freq_save_check_model,
                                       type_save_model='full_model',
                                       update_filename_epoch=True)
        self._list_callbacks.append(new_callback)

        model_filename = join_path_names(models_path, NAME_SAVEDMODEL_LAST_TORCH)
        new_callback = ModelCheckpoint(model_filename, self,
                                       type_save_model='full_model')
        self._list_callbacks.append(new_callback)

        self.freq_validate_model = freq_validate_model

    def summary_model(self) -> None:
        summary(self._network, tuple(self._network.get_size_input()))

    def load_model_only_weights(self, model_filename: str) -> None:
        model_full = torch.load(model_filename, map_location=self._device)
        self._network.load_state_dict(model_full)

    def load_model_full(self, model_filename: str, **kwargs) -> None:
        model_full = torch.load(model_filename, map_location=self._device)

        # create network
        type_network = model_full['network_desc'][0]
        network_input_args = model_full['network_desc'][1]

        update_net_input_args = kwargs['update_net_input_args'] if 'update_net_input_args' in kwargs.keys() else None
        if update_net_input_args:
            network_input_args.update(update_net_input_args)

        self.create_network(type_network, **network_input_args)
        self._network.load_state_dict(model_full['network_state_dict'])

        # create optimizer
        type_optimizer = model_full['optimizer_desc']
        self.create_optimizer(type_optimizer, learn_rate=0.0)
        self._optimizer.load_state_dict(model_full['optimizer_state_dict'])

        # create loss
        type_loss = model_full['loss_desc'][0]
        loss_input_args = model_full['loss_desc'][1]
        self.create_loss(type_loss, is_mask_to_region_interest=loss_input_args['is_masks_exclude'])

        # create list of metrics
        list_type_metrics = model_full['metrics_desc']
        self.create_list_metrics(list_type_metrics, is_mask_to_region_interest=loss_input_args['is_masks_exclude'])

    def save_model_only_weights(self, model_filename: str) -> None:
        torch.save(self._network.state_dict(), model_filename)

    def save_model_full(self, model_filename: str) -> None:
        model_full = {'network_desc': self._network.get_network_input_args(),
                      'network_state_dict': self._network.state_dict(),
                      'optimizer_desc': self._optimizer.__name__,
                      'optimizer_state_dict': self._optimizer.state_dict(),
                      'loss_desc': [self._loss.__class__.__name__, {'is_masks_exclude': self._loss._is_mask_exclude}],
                      'metrics_desc': [imetric.__class__.__name__ for imetric in self._list_metrics]}
        torch.save(model_full, model_filename)

    def _criterion(self, in_prediction: torch.FloatTensor, in_groundtruth: torch.FloatTensor) -> torch.FloatTensor:
        return self._loss.forward(in_groundtruth, in_prediction)

    def _compute_list_metrics(self, in_prediction: torch.FloatTensor, in_groundtruth: torch.FloatTensor) -> List[float]:
        out_list_metrics = []
        for imetric_fun in self._list_metrics:
            out_metric = imetric_fun.compute(in_groundtruth, in_prediction)
            out_list_metrics.append(out_metric.item())

        return out_list_metrics

    def _run_callbacks_on_train_begin(self) -> None:
        for icallback in self._list_callbacks:
            icallback.on_train_begin()

    def _run_callbacks_on_epoch_end(self) -> None:
        for icallback in self._list_callbacks:
            icallback.on_epoch_end(self._epoch_count, self._data_output)


    def train(self,
              train_data_loader: BatchDataGenerator,
              valid_data_loader: BatchDataGenerator = None,
              num_epochs: int = 1,
              max_steps_epoch: int = None,
              initial_epoch: int = 0
              ) -> None:
        self._train_data_loader = train_data_loader
        self._valid_data_loader = valid_data_loader
        self._num_epochs = num_epochs
        self._max_steps_epoch = max_steps_epoch

        self._epoch_count = initial_epoch
        self._epoch_start_count = 0

        self._train_loss = 0.0
        self._valid_loss = 0.0
        self._train_metrics = [0] * self._num_epochs
        self._valid_metrics = [0] * self._num_epochs

        self._networks.to(self._device)     # if 'cuda:0', dispatch model to 'gpu'

        for i_epoch in range(initial_epoch, num_epochs):
            self._run_epoch()
            self._epoch_count += 1
            self._epoch_start_count += 1


    def predict(self, test_data_loader: BatchDataGenerator) -> np.ndarray:
        self._test_data_loader = test_data_loader

        self._networks.to(self._device)     # if 'cuda:0', dispatch model to 'gpu'

        self._networks = self._networks.eval()  # switch to evaluate mode
        self._networks.preprocess(-1)

        output_prediction = self._run_prediction()
        return output_prediction


    def _run_epoch(self) -> None:
        # Run a train and validation pass on the current epoch

        self._networks = self._networks.train()     # switch to train mode
        self._networks.preprocess(self._epoch_count)

        if self._epoch_count == 0:
            self._run_callbacks_on_epoch_end()

        if self._num_metrics > 0:
            (self._train_loss, self._train_metrics) = self._train_epoch()
        else:
            self._train_loss = self._train_epoch()

        if self._valid_data_loader and \
            (self._epoch_count % self.freq_validate_model == 0 or self._epoch_start_count == 0):

            self._networks = self._networks.eval()  # switch to evaluate mode

            if self._num_metrics > 0:
                (self._valid_loss, self._valid_metrics) = self._validation_epoch()
            else:
                self._valid_loss = self._validation_epoch()

        if self._valid_data_loader:
            self._data_output = [self._train_loss, self._valid_loss] + self._train_metrics + self._valid_metrics
        else:
            self._data_output = [self._train_loss] + self._train_metrics

        self._run_callbacks_on_epoch_end()

        # write loss history
        # print("\ntrain loss = {0:.3f}".format(self.train_loss))
        # if self.valid_data_generator:
        # print("valid loss = {0:.3f}".format(self.valid_loss))


    def _train_epoch(self) -> Union[float, Tuple[float, List[float]]]:
        if self._max_steps_epoch and self._max_steps_epoch < len(self._train_data_loader):
            num_batches = self._max_steps_epoch
        else:
            num_batches = len(self._train_data_loader)

        progressbar = tqdm(total= num_batches,
                           desc= 'Epochs {}/{}'.format(self._epoch_count, self._num_epochs),
                           bar_format= '{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]')

        #time_compute = 0.0
        #time_total_ini = dt.now()
        sumrun_loss = 0.0
        sumrun_metrics = [0.0] * self._num_metrics

        i_batch = 0
        for (in_batch_Xdata, in_batch_Ydata) in self._train_data_loader:
            in_batch_Xdata = in_batch_Xdata.to(self._device)
            in_batch_Ydata = in_batch_Ydata.to(self._device)

            #time_ini = dt.now()

            self._optimizer.zero_grad()
            batch_prediction = self._networks(in_batch_Xdata)
            loss = self._criterion(batch_prediction, in_batch_Ydata)
            loss.backward()             # run backprop
            self._optimizer.step()      # optimize grads one step
            loss.detach()
            sumrun_loss += loss.item()

            metrics_this = self._compute_list_metrics(batch_prediction, in_batch_Ydata)
            sumrun_metrics = [value1 + value2 for (value1, value2) in zip(sumrun_metrics, metrics_this)]

            #time_now = dt.now()
            #time_compute += (time_now - time_ini).seconds

            loss_partial = sumrun_loss / (i_batch+1)
            progressbar.set_postfix(loss='{0:1.5f}'.format(loss_partial))
            progressbar.update(1)

            i_batch += 1
            if i_batch > num_batches:
                break

        #time_now = dt.now()
        #time_total = (time_now - time_total_ini).seconds
        #time_loaddata = time_total - time_compute
        #print("\ntime total = {0:.3f}".format(time_total))
        #print("time loaddata / compute = {0:.3f} / {1:.3f}".format(time_loaddata, time_compute))

        total_loss = sumrun_loss / num_batches

        if self._num_metrics > 0:
            total_metrics = [value / num_batches for value in sumrun_metrics]
            return (total_loss, total_metrics)
        else:
            return total_loss


    def _validation_epoch(self) -> Union[float, Tuple[float, List[float]]]:
        if self._max_steps_epoch and self._max_steps_epoch < len(self._valid_data_loader):
            num_batches = self._max_steps_epoch
        else:
            num_batches = len(self._valid_data_loader)

        progressbar = tqdm(total= num_batches, desc= 'Validation', leave= False)

        #time_compute = 0.0
        #time_total_ini = dt.now()
        sumrun_loss = 0.0
        sumrun_metrics = [0.0] * self._num_metrics

        i_batch = 0
        for (in_batch_Xdata, in_batch_Ydata) in self._valid_data_loader:
            in_batch_Xdata = in_batch_Xdata.to(self._device)
            in_batch_Ydata = in_batch_Ydata.to(self._device)

            #time_ini = dt.now()

            with torch.no_grad():
                batch_prediction = self._networks(in_batch_Xdata)
                loss = self._criterion(batch_prediction, in_batch_Ydata)
                loss.detach()
            sumrun_loss += loss.item()

            metrics_this = self._compute_list_metrics(batch_prediction, in_batch_Ydata)
            sumrun_metrics = [value1 + value2 for (value1, value2) in zip(sumrun_metrics, metrics_this)]

            #time_now = dt.now()
            #time_compute += (time_now - time_ini).seconds

            progressbar.update(1)

            i_batch += 1
            if i_batch > num_batches:
                break

        #time_now = dt.now()
        #time_total = (time_now - time_total_ini).seconds
        #time_loaddata = time_total - time_compute
        #print("\ntime total = {0:.3f}".format(time_total))
        #print("time loaddata / compute = {0:.3f} / {1:.3f}".format(time_loaddata, time_compute))

        total_loss = sumrun_loss / num_batches

        if self._num_metrics > 0:
            total_metrics = [value / num_batches for value in sumrun_metrics]
            return (total_loss, total_metrics)
        else:
            return total_loss


    def _run_prediction(self) -> np.ndarray:
        num_batches = len(self._test_data_loader)
        size_output_batch = self._networks.get_size_output()
        num_classes_out = size_output_batch[0]

        out_shape_prediction = (num_batches, num_classes_out) + size_output_batch[1:]
        output_prediction = np.ndarray(out_shape_prediction, dtype=np.float32)

        progressbar = tqdm(total= num_batches, desc='Prediction')

        #time_compute = 0.0
        #time_total_ini = dt.now()

        for i_batch, (in_batch_Xdata, in_batch_Ydata) in enumerate(self._test_data_loader):
            in_batch_Xdata = in_batch_Xdata.to(self._device)

            #time_ini = dt.now()

            with torch.no_grad():
                batch_prediction = self._networks(in_batch_Xdata)
                batch_prediction.detach()

            output_prediction[i_batch] = batch_prediction.cpu()     # dispatch prediction to 'cpu'

            #time_now = dt.now()
            #time_compute_i = (time_now - time_ini).seconds
            #time_compute += time_compute_i

            progressbar.update(1)

        #time_now = dt.now()
        #time_total = (time_now - time_total_ini).seconds
        #time_loaddata = time_total - time_compute
        #print("\ntime total = {0:.3f}".format(time_total))
        #print("time loaddata / compute = {0:.3f} / {1:.3f}".format(time_loaddata, time_compute))

        # place output channels as last dim of output predictions
        ndim_out = len(output_prediction.shape)
        output_prediction = np.rollaxis(output_prediction, 1, ndim_out)

        return output_prediction