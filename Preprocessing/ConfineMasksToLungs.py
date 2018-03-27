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

    DUMMY_VALUE = -1

    @classmethod
    def compute(cls, masks_airways_array, masks_lungs_array):

        # modify masks to exclude areas outside the lungs
        mod_masks_airways_array = np.where(masks_lungs_array == 0, cls.DUMMY_VALUE, masks_airways_array)

        return mod_masks_airways_array