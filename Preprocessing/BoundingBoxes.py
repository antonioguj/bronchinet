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


class BoundingBoxes(object):

    @staticmethod
    def get_mirror_bounding_box(bounding_box,
                                max_dim=512):
        return ((bounding_box[0][0], bounding_box[0][1]),
                (max_dim - bounding_box[1][1], max_dim - bounding_box[1][0]))

    @staticmethod
    def compute_size_bounding_box(bounding_box):
        return ((bounding_box[0][1] - bounding_box[0][0]),
                (bounding_box[1][1] - bounding_box[1][0]),
                (bounding_box[2][1] - bounding_box[2][0]))

    @staticmethod
    def compute_max_size_bounding_box(bounding_box,
                                      max_bounding_box):
        return (max(bounding_box[0], max_bounding_box[0]),
                max(bounding_box[1], max_bounding_box[1]),
                max(bounding_box[2], max_bounding_box[2]))

    @staticmethod
    def compute_min_size_bounding_box(bounding_box,
                                      min_bounding_box):
        return (min(bounding_box[0], min_bounding_box[0]),
                min(bounding_box[1], min_bounding_box[1]),
                min(bounding_box[2], min_bounding_box[2]))

    @staticmethod
    def compute_coords0_bounding_box(bounding_box):
        return (bounding_box[0][0], bounding_box[1][0], bounding_box[2][0])

    @staticmethod
    def compute_default_bounding_box((size_img_z, size_img_x, size_img_y)):
        return ((0, size_img_z), (0, size_img_x), (0, size_img_y))

    @staticmethod
    def compute_split_bounding_boxes(bounding_box, axis=0):
        if axis == 0:
            half_boundbox_zdim = int((bounding_box[0][1] + bounding_box[0][0]) / 2)
            bounding_box_1 = ((bounding_box[0][0], half_boundbox_zdim),
                              (bounding_box[1][0], bounding_box[1][1]),
                              (bounding_box[2][0], bounding_box[2][1]))
            bounding_box_2 = ((half_boundbox_zdim, bounding_box[0][1]),
                              (bounding_box[1][0], bounding_box[1][1]),
                              (bounding_box[2][0], bounding_box[2][1]))
            return (bounding_box_1, bounding_box_2)
        elif axis == 1:
            half_boundbox_xdim = int((bounding_box[1][1] + bounding_box[1][0]) / 2)
            bounding_box_1 = ((bounding_box[0][0], bounding_box[0][1]),
                              (bounding_box[1][0], half_boundbox_xdim),
                              (bounding_box[2][0], bounding_box[2][1]))
            bounding_box_2 = ((bounding_box[0][0], bounding_box[0][1]),
                              (half_boundbox_xdim, bounding_box[1][1]),
                              (bounding_box[2][0], bounding_box[2][1]))
            return (bounding_box_1, bounding_box_2)
        elif axis == 2:
            half_boundbox_ydim = int((bounding_box[2][1] + bounding_box[2][0]) / 2)
            bounding_box_1 = ((bounding_box[0][0], bounding_box[0][1]),
                              (bounding_box[1][0], bounding_box[1][1]),
                              (bounding_box[2][0], half_boundbox_ydim))
            bounding_box_2 = ((bounding_box[0][0], bounding_box[0][1]),
                              (bounding_box[1][0], bounding_box[1][1]),
                              (half_boundbox_ydim, bounding_box[2][1]))
            return (bounding_box_1, bounding_box_2)
        else:
            return False


    @staticmethod
    def fit_bounding_box_to_image_size(bounding_box,
                                       (size_max_z, size_max_x, size_max_y)):
        return ((max(bounding_box[0][0], 0), min(bounding_box[0][1], size_max_z)),
                (max(bounding_box[1][0], 0), min(bounding_box[1][1], size_max_x)),
                (max(bounding_box[2][0], 0), min(bounding_box[2][1], size_max_y)))

    @staticmethod
    def translate_bounding_box_to_image_size(bounding_box, (size_max_z, size_max_x, size_max_y)):
        translate_X = 0
        translate_Y = 0
        translate_Z = 0
        if (bounding_box[1][0] < 0):
            translate_X = -bounding_box[1][0]
        elif (bounding_box[1][1] > size_max_x):
            translate_X = -(bounding_box[1][1] - size_max_x)
        if (bounding_box[2][0] < 0):
            translate_Y = -bounding_box[2][0]
        elif (bounding_box[2][1] > size_max_y):
            translate_Y = -(bounding_box[2][1] - size_max_y)
        if (size_max_z!=0):
            if (bounding_box[0][0] < 0):
                translate_Z = -bounding_box[0][0]
            elif (bounding_box[0][1] > size_max_z):
                translate_Z = -(bounding_box[0][1] - size_max_z)
        if (translate_X != 0 or
            translate_Y != 0 or
            translate_Z != 0):
            return ((bounding_box[0][0] + translate_Z, bounding_box[0][1] + translate_Z),
                    (bounding_box[1][0] + translate_X, bounding_box[1][1] + translate_X),
                    (bounding_box[2][0] + translate_Y, bounding_box[2][1] + translate_Y))
        else:
            return bounding_box

    @staticmethod
    def is_bounding_box_contained(in_bounding_box,
                                  out_bounding_box):
        return (in_bounding_box[0][0] < out_bounding_box[0][0] or in_bounding_box[0][1] > out_bounding_box[0][1] or
                in_bounding_box[1][0] < out_bounding_box[1][0] or in_bounding_box[1][1] > out_bounding_box[1][1] or
                in_bounding_box[2][0] < out_bounding_box[2][0] or in_bounding_box[2][1] > out_bounding_box[2][1])


    @classmethod
    def compute_bounding_box_centered_bounding_box_2D(cls, in_bounding_box,
                                                      size_out_boundbox,
                                                      size_image= None):
        center = (int(np.float(in_bounding_box[1][0] + in_bounding_box[1][1]) / 2),
                  int(np.float(in_bounding_box[2][0] + in_bounding_box[2][1]) / 2))
        begin_boundbox = (center[0] - size_out_boundbox[0] / 2,
                          center[1] - size_out_boundbox[1] / 2)
        bounding_box = ((in_bounding_box[0][0], in_bounding_box[0][1]),
                        (begin_boundbox[0], begin_boundbox[0] + size_out_boundbox[0]),
                        (begin_boundbox[1], begin_boundbox[1] + size_out_boundbox[1]))
        if size_image:
            bounding_box = cls.translate_bounding_box_to_image_size(bounding_box, size_image)
            return cls.fit_bounding_box_to_image_size(bounding_box, size_image)
        else:
            return bounding_box

    @classmethod
    def compute_bounding_box_centered_bounding_box_3D(cls, in_bounding_box,
                                                      size_out_boundbox,
                                                      size_image= None):
        center = (int(np.float(in_bounding_box[0][0] + in_bounding_box[0][1]) / 2),
                  int(np.float(in_bounding_box[1][0] + in_bounding_box[1][1]) / 2),
                  int(np.float(in_bounding_box[2][0] + in_bounding_box[2][1]) / 2))
        begin_boundbox = (center[0] - size_out_boundbox[0] / 2,
                          center[1] - size_out_boundbox[1] / 2,
                          center[2] - size_out_boundbox[2] / 2)
        bounding_box = ((begin_boundbox[0], begin_boundbox[0] + size_out_boundbox[0]),
                        (begin_boundbox[1], begin_boundbox[1] + size_out_boundbox[1]),
                        (begin_boundbox[2], begin_boundbox[2] + size_out_boundbox[2]))
        if size_image:
            bounding_box = cls.translate_bounding_box_to_image_size(bounding_box, size_image)
            return cls.fit_bounding_box_to_image_size(bounding_box, size_image)
        else:
            return bounding_box

    @classmethod
    def compute_bounding_box_centered_bounding_box_2D_OLD(cls, in_bounding_box,
                                                          size_out_boundbox,
                                                          size_image= None):
        center = (int(np.float(in_bounding_box[1][0] + in_bounding_box[1][1]) / 2),
                  int(np.float(in_bounding_box[2][0] + in_bounding_box[2][1]) / 2))
        half_boundbox = (int(np.float(size_out_boundbox[0]) / 2),
                         int(np.float(size_out_boundbox[1]) / 2))
        bounding_box = ((in_bounding_box[0][0], in_bounding_box[0][1]),
                        (center[0] - half_boundbox[0], center[0] + size_out_boundbox[0] - half_boundbox[0]),
                        (center[1] - half_boundbox[1], center[1] + size_out_boundbox[1] - half_boundbox[1]))
        if size_image:
            bounding_box = cls.translate_bounding_box_to_image_size(bounding_box, (0, size_image[1], size_image[2]))
            return cls.fit_bounding_box_to_image_size(bounding_box, size_image)
        else:
            return bounding_box

    @classmethod
    def compute_bounding_box_centered_bounding_box_3D_OLD(cls, in_bounding_box,
                                                          size_out_boundbox,
                                                          size_image= None):
        center = (int(np.float(in_bounding_box[0][0] + in_bounding_box[0][1]) / 2),
                  int(np.float(in_bounding_box[1][0] + in_bounding_box[1][1]) / 2),
                  int(np.float(in_bounding_box[2][0] + in_bounding_box[2][1]) / 2))
        half_boundbox = (int(np.float(size_out_boundbox[0]) / 2),
                         int(np.float(size_out_boundbox[1]) / 2),
                         int(np.float(size_out_boundbox[2]) / 2))
        bounding_box = ((center[0] - half_boundbox[0], center[0] + size_out_boundbox[0] - half_boundbox[0]),
                        (center[1] - half_boundbox[1], center[1] + size_out_boundbox[1] - half_boundbox[1]),
                        (center[2] - half_boundbox[2], center[2] + size_out_boundbox[2] - half_boundbox[2]))
        if size_image:
            bounding_box = cls.translate_bounding_box_to_image_size(bounding_box, size_image)
            return cls.fit_bounding_box_to_image_size(bounding_box, size_image)
        else:
            return bounding_box


    @classmethod
    def compute_bounding_box_centered_image_2D(cls, size_boundbox,
                                               size_image):
        begin_boundbox = ((size_image[1] - size_boundbox[1]) / 2,
                          (size_image[2] - size_boundbox[2]) / 2)
        bounding_box = ((0, size_image[0]),
                        (begin_boundbox[1], begin_boundbox[1] + size_boundbox[1]),
                        (begin_boundbox[2], begin_boundbox[2] + size_boundbox[2]))
        return bounding_box

    @classmethod
    def compute_bounding_box_centered_image_3D(cls, size_boundbox,
                                               size_image):
        begin_boundbox = ((size_image[0] - size_boundbox[0]) / 2,
                          (size_image[1] - size_boundbox[1]) / 2,
                          (size_image[2] - size_boundbox[2]) / 2)
        bounding_box = ((begin_boundbox[0], begin_boundbox[0] + size_boundbox[0]),
                        (begin_boundbox[1], begin_boundbox[1] + size_boundbox[1]),
                        (begin_boundbox[2], begin_boundbox[2] + size_boundbox[2]))
        return bounding_box
    
    @classmethod
    def compute_bounding_box_centered_image_2D_OLD(cls, size_boundbox,
                                                   size_image):
        center = (int(np.float(size_image[1]) / 2),
                  int(np.float(size_image[2]) / 2))
        half_boundbox = (int(np.float(size_boundbox[1]) / 2),
                         int(np.float(size_boundbox[2]) / 2))
        bounding_box = ((0, size_image[0]),
                        (center[0] - half_boundbox[0], center[0] + size_boundbox[0] - half_boundbox[0]),
                        (center[1] - half_boundbox[1], center[1] + size_boundbox[1] - half_boundbox[1]))
        return cls.fit_bounding_box_to_image_size(bounding_box, (0, size_image[1], size_image[2]))

    @classmethod
    def compute_bounding_box_centered_image_3D_OLD(cls, size_boundbox,
                                                   size_image):
        center = (int(np.float(size_image[0]) / 2),
                  int(np.float(size_image[1]) / 2),
                  int(np.float(size_image[2]) / 2))
        half_boundbox = (int(np.float(size_boundbox[0]) / 2),
                         int(np.float(size_boundbox[1]) / 2),
                         int(np.float(size_boundbox[2]) / 2))
        bounding_box = ((center[0] - half_boundbox[0], center[0] + size_boundbox[0] - half_boundbox[0]),
                            (center[1] - half_boundbox[1], center[1] + size_boundbox[1] - half_boundbox[1]),
                            (center[2] - half_boundbox[2], center[2] + size_boundbox[2] - half_boundbox[2]))
        return cls.fit_bounding_box_to_image_size(bounding_box, size_image)


    @classmethod
    def compute_bounding_box_contain_masks_2D(cls, masks_array):
        # find where there are active masks. Elsewhere not interesting
        indexes_active_masks = np.argwhere(masks_array != 0)
        bounding_box = ((0, masks_array.shape[0]),
                        (min(indexes_active_masks[:,1]), max(indexes_active_masks[:,1])),
                        (min(indexes_active_masks[:,2]), max(indexes_active_masks[:,2])))
        return bounding_box

    @classmethod
    def compute_bounding_box_contain_masks_3D(cls, masks_array):
        # find where there are active masks. Elsewhere not interesting
        indexes_active_masks = np.argwhere(masks_array != 0)
        bounding_box = ((min(indexes_active_masks[:,0]), max(indexes_active_masks[:,0])),
                        (min(indexes_active_masks[:,1]), max(indexes_active_masks[:,1])),
                        (min(indexes_active_masks[:,2]), max(indexes_active_masks[:,2])))
        return bounding_box


    @classmethod
    def compute_bounding_box_contain_masks_with_border_effects_2D(cls, masks_array,
                                                                  voxels_buffer_border=BORDER_EFFECTS):
        bounding_box = cls.compute_bounding_box_contain_masks_2D(masks_array)
        # account for border effects
        bounding_box = ((0, masks_array.shape[0]),
                        (bounding_box[1][0] - voxels_buffer_border[2], bounding_box[1][1] + voxels_buffer_border[2]),
                        (bounding_box[2][0] - voxels_buffer_border[3], bounding_box[2][1] + voxels_buffer_border[3]))
        return cls.fit_bounding_box_to_image_size(bounding_box, masks_array.shape)

    @classmethod
    def compute_bounding_box_contain_masks_with_border_effects_3D(cls, masks_array,
                                                                  voxels_buffer_border=BORDER_EFFECTS):
        bounding_box = cls.compute_bounding_box_contain_masks_3D(masks_array)
        # account for border effects
        bounding_box = ((bounding_box[0][0] - voxels_buffer_border[0], bounding_box[0][1] + voxels_buffer_border[1]),
                        (bounding_box[1][0] - voxels_buffer_border[2], bounding_box[1][1] + voxels_buffer_border[2]),
                        (bounding_box[2][0] - voxels_buffer_border[3], bounding_box[2][1] + voxels_buffer_border[3]))
        return cls.fit_bounding_box_to_image_size(bounding_box, masks_array.shape)
