
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


def get_sgd(learn_rate: float, **kwargs) -> SGD_keras:
    return SGD_keras(lr=learn_rate)


def get_sgdmom(learn_rate: float, momentum: float = 0.9, **kwargs) -> SGD_keras:
    return SGD_keras(lr=learn_rate, momentum=momentum)


def get_adagrad(learn_rate: float, **kwargs) -> Adagrad_keras:
    return Adagrad_keras(lr=learn_rate)


def get_rmsprop(learn_rate: float, **kwargs) -> RMSprop_keras:
    return RMSprop_keras(lr=learn_rate)


def get_adadelta(learn_rate: float, **kwargs) -> Adadelta_keras:
    return Adadelta_keras(lr=learn_rate)


def get_adam(learn_rate: float, **kwargs) -> Adam_keras:
    return Adam_keras(lr=learn_rate)
