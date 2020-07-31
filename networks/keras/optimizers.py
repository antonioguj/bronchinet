
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam

from common.exception_manager import catch_error_exception


def get_optimizer_keras(option: str, lr: float):
    list_avail_optimizers = ['SGD',
                             'SGDmom',
                             'Adagrad',
                             'RMSprop',
                             'Adadelta',
                             'Adam',
                             ]
    if option == 'SGD':
        return SGD(lr=lr)
    elif option == 'SGDmom':
        return SGD(lr=lr, momentum=0.9)
    elif option == 'Adagrad':
        return Adagrad(lr=lr)
    elif option == 'RMSprop':
        return RMSprop(lr=lr)
    elif option == 'Adadelta':
        return Adadelta(lr=lr)
    elif option == 'Adam':
        return Adam(lr=lr)
    else:
        message = 'Optimizer chosen not found. Optimizers available: %s' % (', '.join(list_avail_optimizers))
        catch_error_exception(message)


