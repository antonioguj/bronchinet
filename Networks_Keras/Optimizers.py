#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from tensorflow.python.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
from Common.ErrorMessages import *


# all available optimizers
def DICTAVAILOPTIMIZERS(option, lr):
    list_optimizer_avail = ['SGD', 'SGDmom', 'Adagrad', 'RMSprop', 'Adadelta', 'Adam']

    if (option=='SGD'):
        return SGD(lr=lr)
    elif (option=='SGDmom'):
        return SGD(lr=lr, momentum=0.9)
    elif (option=='Adagrad'):
        return Adagrad(lr=lr)
    elif (option=='RMSprop'):
        return RMSprop(lr=lr)
    elif (option=='Adadelta'):
        return Adadelta(lr=lr)
    elif (option=='Adam'):
        return Adam(lr=lr)
    else:
        message = 'Optimizer chosen not found. Optimizers available: (%s)' % (', '.join(list_optimizer_avail))
        CatchErrorException(message)
        return NotImplemented


