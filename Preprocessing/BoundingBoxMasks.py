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

BORDER_EFFECTS = (0, 0, 0)


class BoundingBoxMasks(object):

    @staticmethod
    def getMirrorBoundingBox(boundingBox):

        return ((boundingBox[0][0], boundingBox[0][1]),
                (512 - boundingBox[1][1], 512 - boundingBox[1][0]))

    @staticmethod
    def computeSizeBoundingBox(boundingBox):

        return ((boundingBox[0][1] - boundingBox[0][0]),
                (boundingBox[1][1] - boundingBox[1][0]),
                (boundingBox[2][1] - boundingBox[2][0]))

    @staticmethod
    def computeMaxSizeBoundingBox(boundingBox, max_boundingBox):

        return (max(boundingBox[0], max_boundingBox[0]),
                max(boundingBox[1], max_boundingBox[1]),
                max(boundingBox[2], max_boundingBox[2]))

    @staticmethod
    def computeCoords0BoundingBox(boundingBox):

        return (boundingBox[0][0], boundingBox[1][0], boundingBox[2][0])

    @staticmethod
    def fitBoundingBoxToImageMaxSize(boundingBox, (size_img_x, size_img_y)):

        translate_X = 0
        translate_Y = 0

        if (boundingBox[1][0] < 0):
            translate_X = -boundingBox[1][0]
        elif (boundingBox[1][1] > size_img_x):
            translate_X = -(boundingBox[1][1] - size_img_x)

        if (boundingBox[2][0] < 0):
            translate_Y = -boundingBox[2][0]
        elif (boundingBox[2][1] > size_img_y):
            translate_Y = -(boundingBox[2][1] - size_img_y)

        if (translate_X != 0 or
            translate_Y != 0):
            return ((boundingBox[0][0], boundingBox[0][1]),
                    (boundingBox[1][0] + translate_X, boundingBox[1][1] + translate_X),
                    (boundingBox[2][0] + translate_Y, boundingBox[2][1] + translate_Y))
        else:
            return boundingBox

    @staticmethod
    def isBoundingBoxContained(orig_boundingBox, new_boundingBox):

        return (orig_boundingBox[0][0] < new_boundingBox[0][0] or orig_boundingBox[0][1] > new_boundingBox[0][1] or
                orig_boundingBox[1][0] < new_boundingBox[1][0] or orig_boundingBox[1][1] > new_boundingBox[1][1] or
                orig_boundingBox[2][0] < new_boundingBox[2][0] or orig_boundingBox[2][1] > new_boundingBox[2][1])

    @classmethod
    def computeCenteredBoundingBox(cls, orig_boundingBox, size_proc_boundingBox, (size_img_z, size_img_x, size_img_y)):

        center = (np.float(orig_boundingBox[1][1] + orig_boundingBox[1][0]) / 2,
                  np.float(orig_boundingBox[2][1] + orig_boundingBox[2][0]) / 2)

        new_boundingBox = ((orig_boundingBox[0][0], orig_boundingBox[0][1]),
                           (int(center[0] - size_proc_boundingBox[0] / 2), int(center[0] + size_proc_boundingBox[0] / 2)),
                           (int(center[1] - size_proc_boundingBox[1] / 2), int(center[1] + size_proc_boundingBox[1] / 2)))

        return cls.fitBoundingBoxToImageMaxSize(new_boundingBox, (size_img_x, size_img_y))


    @classmethod
    def compute(cls, masks_array):

        # Find where there are active masks. Elsewhere not interesting
        indexesActiveMasks = np.argwhere(masks_array != 0)

        return ((min(indexesActiveMasks[:,0]), max(indexesActiveMasks[:,0])),
                (min(indexesActiveMasks[:,1]), max(indexesActiveMasks[:,1])),
                (min(indexesActiveMasks[:,2]), max(indexesActiveMasks[:,2])))

    @classmethod
    def compute_with_border_effects(cls, masks_array, voxels_buffer_border=BORDER_EFFECTS):

        boundingBox = cls.compute(masks_array)

        # account for border effects
        boundingBox = ((boundingBox[0][0] - voxels_buffer_border[0], boundingBox[0][1] + voxels_buffer_border[0]),
                       (boundingBox[1][0] - voxels_buffer_border[1], boundingBox[1][1] + voxels_buffer_border[1]),
                       (boundingBox[2][0] - voxels_buffer_border[2], boundingBox[2][1] + voxels_buffer_border[2]))

        (size_img_z, size_img_x, size_img_y) = masks_array.shape

        return ((max(boundingBox[0][0], 0), min(boundingBox[0][1], size_img_z)),
                (max(boundingBox[1][0], 0), min(boundingBox[1][1], size_img_x)),
                (max(boundingBox[2][0], 0), min(boundingBox[2][1], size_img_y)))