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

BORDER_EFFECTS = (0, 0, 0, 0)


class BoundingBoxMasks(object):

    @staticmethod
    def get_mirror_bounding_box(bounding_box):

        return ((bounding_box[0][0], bounding_box[0][1]),
                (512 - bounding_box[1][1], 512 - bounding_box[1][0]))

    @staticmethod
    def compute_size_bounding_box(bounding_box):

        return ((bounding_box[0][1] - bounding_box[0][0]),
                (bounding_box[1][1] - bounding_box[1][0]),
                (bounding_box[2][1] - bounding_box[2][0]))

    @staticmethod
    def compute_max_size_bounding_box(bounding_box, max_bounding_box):

        return (max(bounding_box[0], max_bounding_box[0]),
                max(bounding_box[1], max_bounding_box[1]),
                max(bounding_box[2], max_bounding_box[2]))

    @staticmethod
    def compute_min_size_bounding_box(bounding_box, min_bounding_box):

        return (min(bounding_box[0], min_bounding_box[0]),
                min(bounding_box[1], min_bounding_box[1]),
                min(bounding_box[2], min_bounding_box[2]))

    @staticmethod
    def compute_coords0_bounding_box(bounding_box):

        return (bounding_box[0][0], bounding_box[1][0], bounding_box[2][0])

    @staticmethod
    def fit_bounding_box_to_image_max_size(bounding_box, size_img_x, size_img_y):

        translate_X = 0
        translate_Y = 0

        if (bounding_box[1][0] < 0):
            translate_X = -bounding_box[1][0]
        elif (bounding_box[1][1] > size_img_x):
            translate_X = -(bounding_box[1][1] - size_img_x)

        if (bounding_box[2][0] < 0):
            translate_Y = -bounding_box[2][0]
        elif (bounding_box[2][1] > size_img_y):
            translate_Y = -(bounding_box[2][1] - size_img_y)

        if (translate_X != 0 or
            translate_Y != 0):
            return ((bounding_box[0][0], bounding_box[0][1]),
                    (bounding_box[1][0] + translate_X, bounding_box[1][1] + translate_X),
                    (bounding_box[2][0] + translate_Y, bounding_box[2][1] + translate_Y))
        else:
            return bounding_box

    @staticmethod
    def is_bounding_box_contained(orig_bounding_box, new_bounding_box):

        return (orig_bounding_box[0][0] < new_bounding_box[0][0] or orig_bounding_box[0][1] > new_bounding_box[0][1] or
                orig_bounding_box[1][0] < new_bounding_box[1][0] or orig_bounding_box[1][1] > new_bounding_box[1][1] or
                orig_bounding_box[2][0] < new_bounding_box[2][0] or orig_bounding_box[2][1] > new_bounding_box[2][1])

    @classmethod
    def compute_centered_bounding_box(cls, orig_bounding_box, size_proc_bounding_box, (size_img_z, size_img_x, size_img_y)):

        center = (int(np.float(orig_bounding_box[1][1] + orig_bounding_box[1][0]) / 2),
                  int(np.float(orig_bounding_box[2][1] + orig_bounding_box[2][0]) / 2))

        new_bounding_box = ((orig_bounding_box[0][0], orig_bounding_box[0][1]),
                           (int(center[0] - size_proc_bounding_box[0] / 2), int(center[0] + size_proc_bounding_box[0] / 2)),
                           (int(center[1] - size_proc_bounding_box[1] / 2), int(center[1] + size_proc_bounding_box[1] / 2)))

        return cls.fit_bounding_box_to_image_max_size(new_bounding_box, size_img_x, size_img_y)

    @classmethod
    def compute_centered_bounding_box_type2(cls, size_proc_bounding_box, (size_img_z, size_img_x, size_img_y)):

        center = (int(np.float(size_proc_bounding_box[0]) / 2),
                  int(np.float(size_proc_bounding_box[1]) / 2))

        half_boundary_box = (int(np.float(size_img_x) / 2),
                             int(np.float(size_img_y) / 2))

        new_bounding_box = ((0, size_img_z),
                           (center[0] - half_boundary_box[0], center[0] + size_img_x - half_boundary_box[0]),
                           (center[1] - half_boundary_box[1], center[1] + size_img_y - half_boundary_box[1]))

        return new_bounding_box

    @classmethod
    def compute_centered_bounding_box_type3(cls, size_bounding_box, (size_img_z, size_img_x, size_img_y)):

        center = (int(np.float(size_img_z) / 2),
                  int(np.float(size_img_x) / 2),
                  int(np.float(size_img_y) / 2))

        half_boundary_box = (int(np.float(size_bounding_box[0]) / 2),
                             int(np.float(size_bounding_box[1]) / 2),
                             int(np.float(size_bounding_box[2]) / 2))

        new_bounding_box = ((center[0] - half_boundary_box[0], center[0] + half_boundary_box[0]),
                            (center[1] - half_boundary_box[1], center[1] + half_boundary_box[1]),
                            (center[2] - half_boundary_box[2], center[2] + half_boundary_box[2]))

        return new_bounding_box


    @classmethod
    def compute(cls, masks_array):

        # Find where there are active masks. Elsewhere not interesting
        indexesActiveMasks = np.argwhere(masks_array != 0)

        return ((min(indexesActiveMasks[:,0]), max(indexesActiveMasks[:,0])),
                (min(indexesActiveMasks[:,1]), max(indexesActiveMasks[:,1])),
                (min(indexesActiveMasks[:,2]), max(indexesActiveMasks[:,2])))

    @classmethod
    def compute_with_border_effects(cls, masks_array, voxels_buffer_border=BORDER_EFFECTS):

        bounding_box = cls.compute(masks_array)

        # account for border effects
        bounding_box = ((bounding_box[0][0] - voxels_buffer_border[0], bounding_box[0][1] + voxels_buffer_border[1]),
                        (bounding_box[1][0] - voxels_buffer_border[2], bounding_box[1][1] + voxels_buffer_border[2]),
                        (bounding_box[2][0] - voxels_buffer_border[3], bounding_box[2][1] + voxels_buffer_border[3]))

        (size_img_z, size_img_x, size_img_y) = masks_array.shape

        return ((max(bounding_box[0][0], 0), min(bounding_box[0][1], size_img_z)),
                (max(bounding_box[1][0], 0), min(bounding_box[1][1], size_img_x)),
                (max(bounding_box[2][0], 0), min(bounding_box[2][1], size_img_y)))