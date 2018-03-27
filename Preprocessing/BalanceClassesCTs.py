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


class BalanceClassesCTs(object):

    @classmethod
    def compute(cls, masks_array):

        num_posclass = len(np.argwhere(masks_array > 0))
        num_negclass = masks_array.size - num_posclass

        return (num_posclass, num_negclass)