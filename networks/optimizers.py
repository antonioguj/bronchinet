
from common.constant import TYPE_DNNLIB_USED
from common.exception_manager import catch_error_exception
if TYPE_DNNLIB_USED == 'Pytorch':
    from networks.pytorch.optimizers import SGD, SGD_mom, RMSprop, Adagrad, Adadelta, Adam
elif TYPE_DNNLIB_USED == 'Keras':
    from networks.keras.optimizers import SGD, SGD_mom, RMSprop, Adagrad, Adadelta, Adam

LIST_AVAIL_OPTIMIZERS = ['SGD',
                         'SGD_mom',
                         'Adagrad',
                         'RMSprop',
                         'Adadelta',
                         'Adam',
                         ]

def get_optimizer(type_optimizer: str, lr: float, **kwargs):
    if type_optimizer == 'SGD':
        SGD(lr, **kwargs)
    elif type_optimizer == 'SGD_mom':
        SGD_mom(lr, **kwargs)
    elif type_optimizer == 'Adagrad':
        Adagrad(lr, **kwargs)
    elif type_optimizer == 'RMSprop':
        RMSprop(lr, **kwargs)
    elif type_optimizer == 'Adadelta':
        Adadelta(lr, **kwargs)
    elif type_optimizer == 'Adam':
        Adam(lr, **kwargs)
    else:
        message = 'Choice optimizer not found. Optimizers available: %s' % (', '.join(LIST_AVAIL_OPTIMIZERS))
        catch_error_exception(message)