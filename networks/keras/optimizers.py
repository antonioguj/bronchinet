
from tensorflow.keras.optimizers import SGD as SGD_keras, \
                                        Adagrad as Adagrad_keras, \
                                        RMSprop as RMSprop_keras, \
                                        Adadelta as Adadelta_keras, \
                                        Adam as Adam_keras

def SGD(lr: float, **kwargs):
    return SGD_keras(lr=lr)

def SGD_mom(lr: float, momentum: float = 0.9, **kwargs):
    return SGD_keras(lr=lr, momentum=momentum)

def Adagrad(lr: float, **kwargs):
    return Adagrad_keras(lr=lr)

def RMSprop(lr: float, **kwargs):
    return RMSprop_keras(lr=lr)

def Adadelta(lr: float, **kwargs):
    return Adadelta_keras(lr=lr)

def Adam(lr: float, **kwargs):
    return Adam_keras(lr=lr)