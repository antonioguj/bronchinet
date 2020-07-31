
from torch.optim import SGD as SGD_torch, \
                        Adagrad as Adagrad_torch, \
                        RMSprop as RMSprop_torch, \
                        Adadelta as Adadelta_torch, \
                        Adam as Adam_torch

def SGD(lr: float, **kwargs):
    model_params = kwargs['model_params']
    return SGD_torch(model_params, lr=lr)

def SGD_mom(lr: float, momentum: float = 0.9, **kwargs):
    model_params = kwargs['model_params']
    return SGD_torch(model_params, lr=lr, momentum=momentum)

def Adagrad(lr: float, **kwargs):
    model_params = kwargs['model_params']
    return Adagrad_torch(model_params, lr=lr)

def RMSprop(lr: float, **kwargs):
    model_params = kwargs['model_params']
    return RMSprop_torch(model_params, lr=lr)

def Adadelta(lr: float, **kwargs):
    model_params = kwargs['model_params']
    return Adadelta_torch(model_params, lr=lr)

def Adam(lr: float, **kwargs):
    model_params = kwargs['model_params']
    return Adam_torch(model_params, lr=lr)