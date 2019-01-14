#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from torch.optim import SGD, RMSprop, Adagrad, Adadelta, Adam


# all available optimizers
def DICTAVAILOPTIMIZERS(option, model_params, lr):
    if (option=='SGD'):
        return SGD(model_params, lr=lr)
    elif (option=='SGDmom'):
        return SGD(model_params, lr=lr, momentum=0.9)
    elif (option=='Adagrad'):
        return Adagrad(model_params, lr=lr)
    elif (option=='RMSprop'):
        return RMSprop(model_params, lr=lr)
    elif (option=='Adadelta'):
        return Adadelta(model_params, lr=lr)
    elif (option=='Adam'):
        return Adam(model_params, lr=lr)
    else:
        return 0
