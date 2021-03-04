
from common.exceptionmanager import catch_warning_exception, catch_error_exception
import tensorflow as tf
#from tensorflow.keras.client import device_lib
import multiprocessing
import os


# GPUs installed in personal computer
LARGER_NAME_GPU_INSTALLED = 'Quadro P6000 - gpu1'
SMALLER_NAME_GPU_INSTALLED = 'Quadro P6000 - gpu2'
#SMALLER_NAME_GPU_INSTALLED = 'GeForce GTX 1070'
LARGER_IDDEVICE_GPU_INSTALLED = '0'
SMALLER_IDDEVICE_GPU_INSTALLED = '1'


class DeviceManagerCPU:

    @staticmethod
    def get_device_info():
        all_devices_info = device_lib.list_local_devices()

        prefix_name_device = 'device:CPU'
        for iter_device_info in all_devices_info:
            if prefix_name_device in iter_device_info.name:
                return iter_device_info

        message = "CPU device not found"
        catch_error_exception(message)

    @staticmethod
    def set_session_using_cpu_device():

        config = tf.ConfigProto(device_count={'CPU': 0})
        # Create a session with the above options specified
        session = tf.Session(config=config)
        K.set_session(session)

    @staticmethod
    def get_num_working_cpu_processes():
        return multiprocessing.cpu_count()


class DeviceManagerGPU:

    frac_gpu_usage_default = 0.9

    @staticmethod
    def get_name_gpu(type_gpu_installed):

        if type_gpu_installed == 'larger_GPU':
            return LARGER_NAME_GPU_INSTALLED
        elif type_gpu_installed == 'smaller_GPU':
            return SMALLER_NAME_GPU_INSTALLED
        else:
            return False

    @staticmethod
    def get_id_device_gpu_homemade(type_gpu_installed):

        if type_gpu_installed == 'larger_GPU':
            return LARGER_IDDEVICE_GPU_INSTALLED
        elif type_gpu_installed == 'smaller_GPU':
            return SMALLER_IDDEVICE_GPU_INSTALLED
        else:
            return False

    @staticmethod
    def get_id_device_from_gpu_name(name_gpu):
        all_devices_info = device_lib.list_local_devices()

        prefix_name_device = 'device:GPU'
        for iter_device_info in all_devices_info:
            if prefix_name_device in iter_device_info.name:
                if name_gpu in iter_device_info.physical_device_desc:
                    return int(iter_device_info.name[-1])
        message = "GPU with name '%s' not found" %(name_gpu)
        catch_error_exception(message)

    @staticmethod
    def get_device_info(id_device=0):
        all_devices_info = device_lib.list_local_devices()

        device_name = 'device:GPU:' + str(id_device)
        for iter_device_info in all_devices_info:
            if device_name == iter_device_info.name:
                return iter_device_info
        message = "GPU device '%s' not found" %(id_device)
        catch_error_exception(message)


    @classmethod
    def set_session_using_1gpu_device(cls, type_gpu_installed='larger_GPU'):

        id_device_gpu = cls.get_id_device_gpu_homemade(type_gpu_installed)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(id_device_gpu)

    @classmethod
    def set_session_using_1gpu_device_withTFapi(cls, type_gpu_installed='larger_GPU',
                                                allow_groth_gpu_memory=False,
                                                set_limit_gpu_usage=False,
                                                frac_gpu_usage=frac_gpu_usage_default):

        name_gpu_installed = cls.get_name_gpu(type_gpu_installed)

        id_device_gpu = cls.get_id_device_from_gpu_name(name_gpu=name_gpu_installed)

        #config = tf.ConfigProto(device_count={'GPU': id_device_gpu})
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(id_device_gpu)

        if allow_groth_gpu_memory:
            # don't pre-allocate memory; allocate as-needed
            config.gpu_options.allow_growth = True
        if set_limit_gpu_usage:
            # only allow a total of half the GPU memory to be allocated
            config.gpu_options.per_process_gpu_memory_fraction = frac_gpu_usage

        # create a session with the above options specified
        session = tf.Session(config=config)
        K.set_session(session)


    @classmethod
    def set_session_using_2gpus_device(cls):

        os.environ['CUDA_VISIBLE_DEVICES'] = str('0,1')

    @classmethod
    def set_session_using_2gpus_device_withTFapi(cls, allow_groth_gpu_memory=False,
                                                 set_limit_gpu_usage=False,
                                                 frac_gpu_usage=frac_gpu_usage_default):

        #config = tf.ConfigProto(device_count={'GPU': [0,1]})
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str('0,1')

        if allow_groth_gpu_memory:
            # don't pre-allocate memory; allocate as-needed
            config.gpu_options.allow_growth = True
        if set_limit_gpu_usage:
            # only allow a total of half the GPU memory to be allocated
            config.gpu_options.per_process_gpu_memory_fraction = frac_gpu_usage

        # create a session with the above options specified
        session = tf.Session(config=config)
        K.set_session(session)


def set_session_in_selected_device(use_GPU_device=False,
                                   type_GPU_installed='larger_GPU'):
    if use_GPU_device:
        if type_GPU_installed == 'larger_GPU':
            print('Select Larger GPU installed: %s...' %(LARGER_NAME_GPU_INSTALLED))
        elif type_GPU_installed == 'smaller_GPU':
            print('Select Smaller GPU installed: %s...' %(SMALLER_NAME_GPU_INSTALLED))

        DeviceManagerGPU.set_session_using_1gpu_device(type_gpu_installed=type_GPU_installed)
        #GPUdevicesManager.set_session_using_2gpus_device()
    else:
        DeviceManagerCPU.set_session_using_cpu_device()