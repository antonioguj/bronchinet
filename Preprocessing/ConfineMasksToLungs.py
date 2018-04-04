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


class ConfineMasksToLungs(object):

    val_exclude = 0

    @classmethod
    def compute(cls, masks_airways_array, masks_lungs_array):

        # modify masks to exclude areas outside the lungs
        masks_airways_array = np.where(masks_lungs_array == 0, cls.val_exclude, masks_airways_array)

        return masks_airways_array