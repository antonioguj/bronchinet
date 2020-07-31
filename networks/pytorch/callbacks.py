
from networks.callbacks import Callback

class Callback_Pytorch(Callback):

    def __call__(self, *args, **kwargs):
        raise NotImplementedError