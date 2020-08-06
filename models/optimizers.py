
from common.constant import TYPE_DNNLIB_USED
from common.exceptionmanager import catch_error_exception
if TYPE_DNNLIB_USED == 'Pytorch':
    from models.pytorch.optimizers import SGD, SGD_mom, RMSprop, Adagrad, Adadelta, Adam
elif TYPE_DNNLIB_USED == 'Keras':
    from models.keras.optimizers import SGD, SGD_mom, RMSprop, Adagrad, Adadelta, Adam

LIST_AVAIL_OPTIMIZERS = ['SGD',
                         'SGD_mom',
                         'Adagrad',
                         'RMSprop',
                         'Adadelta',
                         'Adam',
                         ]

def get_optimizer(type_optimizer: str, learn_rate: float, **kwargs):
    if type_optimizer == 'SGD':
        SGD(learn_rate, **kwargs)
    elif type_optimizer == 'SGD_mom':
        SGD_mom(learn_rate, **kwargs)
    elif type_optimizer == 'Adagrad':
        Adagrad(learn_rate, **kwargs)
    elif type_optimizer == 'RMSprop':
        RMSprop(learn_rate, **kwargs)
    elif type_optimizer == 'Adadelta':
        Adadelta(learn_rate, **kwargs)
    elif type_optimizer == 'Adam':
        Adam(learn_rate, **kwargs)
    else:
        message = 'Choice Optimizer not found. Optimizers available: %s' % (', '.join(LIST_AVAIL_OPTIMIZERS))
        catch_error_exception(message)