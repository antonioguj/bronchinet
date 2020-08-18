
from torch.optim import SGD as SGD_torch, \
                        Adagrad as Adagrad_torch, \
                        RMSprop as RMSprop_torch, \
                        Adadelta as Adadelta_torch, \
                        Adam as Adam_torch

LIST_AVAIL_OPTIMIZERS = ['SGD',
                         'SGD_mom',
                         'Adagrad',
                         'RMSprop',
                         'Adadelta',
                         'Adam',
                         ]

def SGD(learn_rate: float, **kwargs):
    model_params = kwargs['model_params']
    return SGD_torch(model_params, lr=learn_rate)

def SGD_mom(learn_rate: float, momentum: float = 0.9, **kwargs):
    model_params = kwargs['model_params']
    return SGD_torch(model_params, lr=learn_rate, momentum=momentum)

def Adagrad(learn_rate: float, **kwargs):
    model_params = kwargs['model_params']
    return Adagrad_torch(model_params, lr=learn_rate)

def RMSprop(learn_rate: float, **kwargs):
    model_params = kwargs['model_params']
    return RMSprop_torch(model_params, lr=learn_rate)

def Adadelta(learn_rate: float, **kwargs):
    model_params = kwargs['model_params']
    return Adadelta_torch(model_params, lr=learn_rate)

def Adam(learn_rate: float, **kwargs):
    model_params = kwargs['model_params']
    return Adam_torch(model_params, lr=learn_rate)