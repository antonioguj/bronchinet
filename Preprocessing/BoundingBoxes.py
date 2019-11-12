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


class BoundingBoxes(object):

    @staticmethod
    def get_mirror_bounding_box(bounding_box,
                                max_dim=512):
        return ((bounding_box[0][0], bounding_box[0][1]),
                (max_dim - bounding_box[1][1], max_dim - bounding_box[1][0]))

    @staticmethod
    def get_size_bounding_box(bounding_box):
        return ((bounding_box[0][1] - bounding_box[0][0]),
                (bounding_box[1][1] - bounding_box[1][0]),
                (bounding_box[2][1] - bounding_box[2][0]))

    @staticmethod
    def get_max_size_bounding_box(bounding_box,
                                  max_bounding_box):
        return (max(bounding_box[0], max_bounding_box[0]),
                max(bounding_box[1], max_bounding_box[1]),
                max(bounding_box[2], max_bounding_box[2]))

    @staticmethod
    def get_min_size_bounding_box(bounding_box,
                                  min_bounding_box):
        return (min(bounding_box[0], min_bounding_box[0]),
                min(bounding_box[1], min_bounding_box[1]),
                min(bounding_box[2], min_bounding_box[2]))

    @staticmethod
    def get_coords0_bounding_box(bounding_box):
        return (bounding_box[0][0], bounding_box[1][0], bounding_box[2][0])

    @staticmethod
    def get_default_bounding_box((size_img_z, size_img_x, size_img_y)):
        return ((0, size_img_z), (0, size_img_x), (0, size_img_y))


    @staticmethod
    def is_bounding_box_contained(in_bounding_box,
                                  out_bounding_box):
        return (in_bounding_box[0][0] < out_bounding_box[0][0] or in_bounding_box[0][1] > out_bounding_box[0][1] or
                in_bounding_box[1][0] < out_bounding_box[1][0] or in_bounding_box[1][1] > out_bounding_box[1][1] or
                in_bounding_box[2][0] < out_bounding_box[2][0] or in_bounding_box[2][1] > out_bounding_box[2][1])

    @staticmethod
    def fit_bounding_box_to_image_size(in_bounding_box, (size_max_z, size_max_x, size_max_y)):
        return ((max(in_bounding_box[0][0], 0), min(in_bounding_box[0][1], size_max_z)),
                (max(in_bounding_box[1][0], 0), min(in_bounding_box[1][1], size_max_x)),
                (max(in_bounding_box[2][0], 0), min(in_bounding_box[2][1], size_max_y)))

    @staticmethod
    def enlarge_bounding_box_borders(in_bounding_box, size_borders_buffer):
        return ((in_bounding_box[0][0] - size_borders_buffer[0], in_bounding_box[0][1] + size_borders_buffer[0]),
                (in_bounding_box[1][0] - size_borders_buffer[1], in_bounding_box[1][1] + size_borders_buffer[1]),
                (in_bounding_box[2][0] - size_borders_buffer[2], in_bounding_box[2][1] + size_borders_buffer[2]))

    @staticmethod
    def translate_bounding_box_to_image_size(in_bounding_box, (size_max_z, size_max_x, size_max_y)):
        shiftX_dist = 0
        shiftY_dist = 0
        shiftZ_dist = 0
        if (in_bounding_box[1][0] < 0):
            shiftX_dist = -in_bounding_box[1][0]
        elif (in_bounding_box[1][1] > size_max_x):
            shiftX_dist = -(in_bounding_box[1][1] - size_max_x)
        if (in_bounding_box[2][0] < 0):
            shiftY_dist = -in_bounding_box[2][0]
        elif (in_bounding_box[2][1] > size_max_y):
            shiftY_dist = -(in_bounding_box[2][1] - size_max_y)
        if (size_max_z!=0):
            if (in_bounding_box[0][0] < 0):
                shiftZ_dist = -in_bounding_box[0][0]
            elif (in_bounding_box[0][1] > size_max_z):
                shiftZ_dist = -(in_bounding_box[0][1] - size_max_z)
        if (shiftX_dist != 0 or
            shiftY_dist != 0 or
            shiftZ_dist != 0):
            return ((in_bounding_box[0][0] + shiftZ_dist, in_bounding_box[0][1] + shiftZ_dist),
                    (in_bounding_box[1][0] + shiftX_dist, in_bounding_box[1][1] + shiftX_dist),
                    (in_bounding_box[2][0] + shiftY_dist, in_bounding_box[2][1] + shiftY_dist))
        else:
            return in_bounding_box

    @staticmethod
    def compute_split_bounding_boxes(in_bounding_box, axis=0):
        if axis == 0:
            half_boundbox_zdim = int((in_bounding_box[0][1] + in_bounding_box[0][0]) / 2)
            out_bounding_box_1 = ((in_bounding_box[0][0], half_boundbox_zdim),
                                  (in_bounding_box[1][0], in_bounding_box[1][1]),
                                  (in_bounding_box[2][0], in_bounding_box[2][1]))
            out_bounding_box_2 = ((half_boundbox_zdim, in_bounding_box[0][1]),
                                  (in_bounding_box[1][0], in_bounding_box[1][1]),
                                  (in_bounding_box[2][0], in_bounding_box[2][1]))
            return (out_bounding_box_1, out_bounding_box_2)
        elif axis == 1:
            half_boundbox_xdim = int((in_bounding_box[1][1] + in_bounding_box[1][0]) / 2)
            out_bounding_box_1 = ((in_bounding_box[0][0], in_bounding_box[0][1]),
                                  (in_bounding_box[1][0], half_boundbox_xdim),
                                  (in_bounding_box[2][0], in_bounding_box[2][1]))
            out_bounding_box_2 = ((in_bounding_box[0][0], in_bounding_box[0][1]),
                                  (half_boundbox_xdim, in_bounding_box[1][1]),
                                  (in_bounding_box[2][0], in_bounding_box[2][1]))
            return (out_bounding_box_1, out_bounding_box_2)
        elif axis == 2:
            half_boundbox_ydim = int((in_bounding_box[2][1] + in_bounding_box[2][0]) / 2)
            out_bounding_box_1 = ((in_bounding_box[0][0], in_bounding_box[0][1]),
                                  (in_bounding_box[1][0], in_bounding_box[1][1]),
                                  (in_bounding_box[2][0], half_boundbox_ydim))
            out_bounding_box_2 = ((in_bounding_box[0][0], in_bounding_box[0][1]),
                                  (in_bounding_box[1][0], in_bounding_box[1][1]),
                                  (half_boundbox_ydim, in_bounding_box[2][1]))
            return (out_bounding_box_1, out_bounding_box_2)
        else:
            return False

    @classmethod
    def compute_bounding_box_contain_masks(cls, in_mask_array,
                                           size_borders_buffer=(0, 0, 0),
                                           is_bounding_box_slices=False):
        # find where there are active masks. Fit bounding box that contain all masks
        indexes_active_masks = np.argwhere(in_mask_array != 0)
        out_bounding_box = ((min(indexes_active_masks[:,0]), max(indexes_active_masks[:,0])),
                            (min(indexes_active_masks[:,1]), max(indexes_active_masks[:,1])),
                            (min(indexes_active_masks[:,2]), max(indexes_active_masks[:,2])))

        # enlarge bounding box to account for border effects
        out_bounding_box = cls.enlarge_bounding_box_borders(out_bounding_box, size_borders_buffer)
        out_bounding_box = cls.fit_bounding_box_to_image_size(out_bounding_box, in_mask_array.shape)

        if is_bounding_box_slices:
            return ((0, in_mask_array.shape[0]),
                    (out_bounding_box[1][0], out_bounding_box[1][1]),
                    (out_bounding_box[2][0], out_bounding_box[2][1]))
        else:
            return out_bounding_box


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
