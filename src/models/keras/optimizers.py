
from tensorflow.keras.optimizers import SGD as SGD_keras, \
    Adagrad as Adagrad_keras, \
    RMSprop as RMSprop_keras, \
    Adadelta as Adadelta_keras, \
    Adam as Adam_keras

LIST_AVAIL_OPTIMIZERS = ['SGD',
                         'SGDmom',
                         'Adagrad',
                         'RMSprop',
                         'Adadelta',
                         'Adam',
                         ]


def get_sgd(learn_rate: float, **kwargs):
    return SGD_keras(lr=learn_rate)


def get_sgdmom(learn_rate: float, momentum: float = 0.9, **kwargs):
    return SGD_keras(lr=learn_rate, momentum=momentum)


def get_adagrad(learn_rate: float, **kwargs):
    return Adagrad_keras(lr=learn_rate)


def get_rmsprop(learn_rate: float, **kwargs):
    return RMSprop_keras(lr=learn_rate)


def get_adadelta(learn_rate: float, **kwargs):
    return Adadelta_keras(lr=learn_rate)


def get_adam(learn_rate: float, **kwargs):
    return Adam_keras(lr=learn_rate)
