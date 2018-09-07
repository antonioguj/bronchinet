#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

import numpy as np


class BalanceClassesMasks(object):

    value_exclude  = -1
    value_foregrnd = 1
    value_backgrnd = 0

    @classmethod
    def compute(cls, masks_array):

        numvox_foregrnd_class = len(np.where(masks_array == cls.value_foregrnd)[0])
        numvox_backgrnd_class = len(np.where(masks_array != cls.value_foregrnd)[0])

        return (numvox_foregrnd_class, numvox_backgrnd_class)

    @classmethod
    def compute_with_exclusion(cls, masks_array):

        numvox_exclude_class  = len(np.where(masks_array == cls.value_exclude )[0])
        numvox_foregrnd_class = len(np.where(masks_array == cls.value_foregrnd)[0])
        numvox_backgrnd_class = len(np.where(masks_array == cls.value_backgrnd)[0])

        return (numvox_foregrnd_class, numvox_backgrnd_class)
