
from typing import Tuple, List
import numpy as np

import torch
from torchsummary import summary
from tqdm import tqdm

from common.constant import NAME_LOSSHISTORY_FILE, NAME_SAVEDMODEL_EPOCH_TORCH, NAME_SAVEDMODEL_LAST_TORCH
from common.functionutil import ImagesUtil, join_path_names
from dataloaders.batchdatagenerator import BatchDataGenerator
from models.modeltrainer import ModelTrainerBase
from models.pytorch.callbacks import RecordLossHistory, ModelCheckpoint


class ModelTrainer(ModelTrainerBase):

    def __init__(self):
        super(ModelTrainer, self).__init__()
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def create_network(self, *args, **kwargs) -> None:
        super(ModelTrainer, self).create_network(*args, **kwargs)

        is_model_half_precision = kwargs['is_model_half_precision'] if 'is_model_half_precision' in kwargs.keys() else None
        if is_model_half_precision:
            self._network.half()

        self._network.to(self._device)  # if 'cuda:0', dispatch model to 'gpu'

    def finalise_model(self) -> None:
        pass

    def create_callbacks(self, models_path: str, **kwargs) -> None:
        self._list_callbacks = []

        is_validation_data = kwargs['is_validation_data'] if 'is_validation_data' in kwargs.keys() else True
        freq_save_check_model = kwargs['freq_save_check_model'] if 'freq_save_check_model' in kwargs.keys() else 1
        freq_validate_model = kwargs['freq_validate_model'] if 'freq_validate_model' in kwargs.keys() else 1
        
        losshistory_filename = join_path_names(models_path, NAME_LOSSHISTORY_FILE)
        new_callback = RecordLossHistory(losshistory_filename, self._list_metrics,
                                         is_hist_validation=is_validation_data)
        self._list_callbacks.append(new_callback)

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
        summary(self._network, self._network.get_size_input())

    def load_model_only_weights(self, model_filename: str) -> None:
        model_state_dict = torch.load(model_filename, map_location=self._device)
        self._network.load_state_dict(model_state_dict)

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

        self.finalise_model()

    def load_model_full_backward_compat(self, model_filename: str, **kwargs) -> None:
        model_full = torch.load(model_filename, map_location=self._device)

        # create network
        type_network = 'UNet3D_Plugin'
        network_input_args_orig = model_full['model_desc'][1]
        # replace the network arguments that were renamed in the new version
        network_input_args = {}
        network_input_args['size_image_in'] = network_input_args_orig['size_image']
        network_input_args['num_levels'] = network_input_args_orig['num_levels'] if 'num_levels' in network_input_args_orig.keys() else 5
        network_input_args['num_featmaps_in'] = network_input_args_orig['num_featmaps_in'] if 'num_featmaps_in' in network_input_args_orig.keys() else 16
        network_input_args['num_channels_in'] = network_input_args_orig['num_channels_in'] if 'num_channels_in' in network_input_args_orig.keys() else 1
        network_input_args['num_classes_out'] = network_input_args_orig['num_classes_out'] if 'num_classes_out' in network_input_args_orig.keys() else 1
        network_input_args['is_use_valid_convols'] = network_input_args_orig['isUse_valid_convols'] if 'isUse_valid_convols' in network_input_args_orig.keys() else False

        update_net_input_args = kwargs['update_net_input_args'] if 'update_net_input_args' in kwargs.keys() else None
        if update_net_input_args:
            network_input_args.update(update_net_input_args)

        self.create_network(type_network, **network_input_args)

        network_state_dict_orig = model_full['model_state_dict']
        # replace the network state class variables that were renamed in the new version
        network_state_dict = {}
        for (key, value) in network_state_dict_orig.items():
            new_key = key.replace('convolution_downlay', '_convolution_down_lev')
            new_key = new_key.replace('convolution_uplay', '_convolution_up_lev')
            new_key = new_key.replace('classification_layer', '_classification_last')
            network_state_dict[new_key] = value

        self._network.load_state_dict(network_state_dict)

        # create optimizer
        type_optimizer = model_full['optimizer_desc']
        self.create_optimizer(type_optimizer, learn_rate=0.0)
        self._optimizer.load_state_dict(model_full['optimizer_state_dict'])

        # create loss
        type_loss = model_full['loss_fun_desc'][0]
        loss_input_args = model_full['loss_fun_desc'][1]
        self.create_loss(type_loss, is_mask_to_region_interest=loss_input_args['is_masks_exclude'])

        self.finalise_model()

    def save_model_only_weights(self, model_filename: str) -> None:
        torch.save(self._network.state_dict(), model_filename)

    def save_model_full(self, model_filename: str) -> None:
        model_full = {'network_desc': [self._network.__class__.__name__, self._network.get_network_input_args()],
                      'network_state_dict': self._network.state_dict(),
                      'optimizer_desc': self._optimizer.__class__.__name__,
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

    def _run_callbacks_on_epoch_end(self, epoch: int, data_output: List[float]) -> None:
        for icallback in self._list_callbacks:
            icallback.on_epoch_end(epoch, data_output)

    def get_size_output_image_model(self):
        return self._network.get_size_output()[1:]


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

        for i_epoch in range(initial_epoch, num_epochs):
            self._run_epoch()
            self._epoch_count += 1
            self._epoch_start_count += 1


    def predict(self, test_data_loader: BatchDataGenerator) -> np.ndarray:
        self._test_data_loader = test_data_loader

        self._network.eval()  # switch to evaluate mode
        self._network.preprocess(-1)

        output_prediction = self._run_prediction()
        return output_prediction


    def _run_epoch(self) -> None:
        # Run a train and validation pass on the current epoch

        self._network.train()     # switch to train mode
        self._network.preprocess(self._epoch_count)

        if self._epoch_count == 0:
            self._run_callbacks_on_train_begin()

        (train_loss, train_metrics) = self._train_epoch()

        if self._valid_data_loader:
            if (self._epoch_count % self.freq_validate_model == 0) or (self._epoch_start_count == 0):

                self._network.eval()  # switch to evaluate mode

                (valid_loss, valid_metrics) = self._validation_epoch()

                self._valid_loss_hold = valid_loss
                self._valid_metrics_hold = valid_metrics
            else:
                valid_loss = self._valid_loss_hold
                valid_metrics = self._valid_metrics_hold
        else:
            valid_loss = 0.0
            valid_metrics = [0.0] * self._num_metrics

        if self._valid_data_loader:
            data_output = [train_loss, valid_loss] + train_metrics + valid_metrics
        else:
            data_output = [train_loss] + train_metrics

        self._run_callbacks_on_epoch_end(self._epoch_count, data_output)

        # write loss history
        # print("\ntrain loss = {0:.3f}".format(self.train_loss))
        # if self.valid_data_generator:
        # print("valid loss = {0:.3f}".format(self.valid_loss))


    def _train_epoch(self) -> Tuple[float, List[float]]:
        if self._max_steps_epoch and self._max_steps_epoch < len(self._train_data_loader):
            num_batches = self._max_steps_epoch
        else:
            num_batches = len(self._train_data_loader)

        progressbar = tqdm(total= num_batches,
                           desc= 'Epochs {}/{}'.format(self._epoch_count+1, self._num_epochs),
                           bar_format= '{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]')

        #time_compute = 0.0
        #time_total_ini = dt.now()
        sumrun_loss = 0.0
        sumrun_metrics = [0.0] * self._num_metrics

        i_batch = 0
        for (in_batch_Xdata, in_batch_Ydata) in self._train_data_loader:
            in_batch_Xdata.to(self._device)
            in_batch_Ydata.to(self._device)

            #time_ini = dt.now()

            self._optimizer.zero_grad()
            batch_prediction = self._network(in_batch_Xdata)
            loss = self._criterion(batch_prediction, in_batch_Ydata)
            loss.backward()             # run backprop
            self._optimizer.step()      # optimize grads one step
            loss.detach()
            sumrun_loss += loss.item()

            metrics_this = self._compute_list_metrics(batch_prediction, in_batch_Ydata)
            sumrun_metrics = [val1 + val2 for (val1, val2) in zip(sumrun_metrics, metrics_this)]

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

        total_loss = sumrun_loss / float(num_batches)
        total_metrics = [value / float(num_batches) for value in sumrun_metrics]

        return (total_loss, total_metrics)


    def _validation_epoch(self) -> Tuple[float, List[float]]:
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
            in_batch_Xdata.to(self._device)
            in_batch_Ydata.to(self._device)

            #time_ini = dt.now()

            with torch.no_grad():
                batch_prediction = self._network(in_batch_Xdata)
                loss = self._criterion(batch_prediction, in_batch_Ydata)
                loss.detach()
            sumrun_loss += loss.item()

            metrics_this = self._compute_list_metrics(batch_prediction, in_batch_Ydata)
            sumrun_metrics = [val1 + val2 for (val1, val2) in zip(sumrun_metrics, metrics_this)]

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

        total_loss = sumrun_loss / float(num_batches)
        total_metrics = [value / float(num_batches) for value in sumrun_metrics]

        return (total_loss, total_metrics)


    def _run_prediction(self) -> np.ndarray:
        num_batches = len(self._test_data_loader)
        size_output_batch = self._network.get_size_output()
        out_shape_prediction = (num_batches,) + size_output_batch
        output_prediction = np.ndarray(out_shape_prediction, dtype=np.float32)

        progressbar = tqdm(total= num_batches, desc='Prediction')

        #time_compute = 0.0
        #time_total_ini = dt.now()

        for i_batch, in_batch_Xdata in enumerate(self._test_data_loader):
            in_batch_Xdata.to(self._device)

            #time_ini = dt.now()

            with torch.no_grad():
                batch_prediction = self._network(in_batch_Xdata)
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

        return ImagesUtil.reshape_channels_last(output_prediction)  # output format "channels_last"