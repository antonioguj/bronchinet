
from tensorflow.keras.optimizers import SGD as SGD_keras, \
                                        Adagrad as Adagrad_keras, \
                                        RMSprop as RMSprop_keras, \
                                        Adadelta as Adadelta_keras, \
                                        Adam as Adam_keras

def SGD(learn_rate: float, **kwargs):
    return SGD_keras(lr=learn_rate)

def SGD_mom(learn_rate: float, momentum: float = 0.9, **kwargs):
    return SGD_keras(lr=learn_rate, momentum=momentum)

def Adagrad(learn_rate: float, **kwargs):
    return Adagrad_keras(lr=learn_rate)

def RMSprop(learn_rate: float, **kwargs):
    return RMSprop_keras(lr=learn_rate)

def Adadelta(learn_rate: float, **kwargs):
    return Adadelta_keras(lr=learn_rate)

def Adam(learn_rate: float, **kwargs):
    return Adam_keras(lr=learn_rate)