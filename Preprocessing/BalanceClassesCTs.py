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

    val_exclude = -1

    @staticmethod
    def compute(masks_array):

        num_pos_class = len(np.where(masks_array != 0)[0])
        num_neg_class = len(np.where(masks_array == 0)[0])

        return (num_pos_class, num_neg_class)

    @classmethod
    def compute_excludeAreas(cls, masks_array):

        num_pos_class = len(np.where(np.logical_and(masks_array != 0, masks_array != cls.val_exclude))[0])
        num_neg_class = len(np.where(masks_array == 0)[0])

        return (num_pos_class, num_neg_class)
