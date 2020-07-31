
from torch.optim import SGD, RMSprop, Adagrad, Adadelta, Adam

from common.exception_manager import catch_error_exception


def get_optimizer_pytorch(option: str, model_params, lr: float):
    list_avail_optimizers = ['SGD',
                             'SGDmom',
                             'Adagrad',
                             'RMSprop',
                             'Adadelta',
                             'Adam',
                             ]
    if option == 'SGD':
        return SGD(model_params, lr= lr)
    elif option == 'SGDmom':
        return SGD(model_params, lr= lr, momentum= 0.9)
    elif option == 'Adagrad':
        return Adagrad(model_params, lr= lr)
    elif option == 'RMSprop':
        return RMSprop(model_params, lr= lr)
    elif option == 'Adadelta':
        return Adadelta(model_params, lr= lr)
    elif option == 'Adam':
        return Adam(model_params, lr= lr)
    else:
        message = 'Optimizer chosen not found. Optimizers available: %s' % (', '.join(list_avail_optimizers))
        catch_error_exception(message)