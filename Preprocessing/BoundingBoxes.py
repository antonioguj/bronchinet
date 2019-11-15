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
    def get_max_size_bounding_box(bounding_box, max_bounding_box):
        return (max(bounding_box[0], max_bounding_box[0]),
                max(bounding_box[1], max_bounding_box[1]),
                max(bounding_box[2], max_bounding_box[2]))

    @staticmethod
    def get_min_size_bounding_box(bounding_box, min_bounding_box):
        return (min(bounding_box[0], min_bounding_box[0]),
                min(bounding_box[1], min_bounding_box[1]),
                min(bounding_box[2], min_bounding_box[2]))

    @staticmethod
    def get_coords0_bounding_box(bounding_box):
        return (bounding_box[0][0], bounding_box[1][0], bounding_box[2][0])

    @staticmethod
    def get_center_bounding_box(bounding_box):
        return ((bounding_box[0][0] + bounding_box[0][1]) / 2,
                (bounding_box[1][0] + bounding_box[1][1]) / 2,
                (bounding_box[2][0] + bounding_box[2][1]) / 2)

    @staticmethod
    def get_create_bounding_box(center_boundbox, size_bounding_box):
        begin_bounding_box = (center_boundbox[0] - size_bounding_box[0] / 2,
                              center_boundbox[1] - size_bounding_box[1] / 2,
                              center_boundbox[2] - size_bounding_box[2] / 2)
        return ((begin_bounding_box[0], begin_bounding_box[0] + size_bounding_box[0]),
                (begin_bounding_box[1], begin_bounding_box[1] + size_bounding_box[1]),
                (begin_bounding_box[2], begin_bounding_box[2] + size_bounding_box[2]))

    @staticmethod
    def get_default_bounding_box_image(size_image):
        return ((0, size_image[0]), (0, size_image[1]), (0, size_image[2]))

    @staticmethod
    def is_bounding_box_contained_in_bounding_box(in_bounding_box, ref_bounding_box):
        return (in_bounding_box[0][0] > ref_bounding_box[0][0] and in_bounding_box[0][1] < ref_bounding_box[0][1] and
                in_bounding_box[1][0] > ref_bounding_box[1][0] and in_bounding_box[1][1] < ref_bounding_box[1][1] and
                in_bounding_box[2][0] > ref_bounding_box[2][0] and in_bounding_box[2][1] < ref_bounding_box[2][1])

    @staticmethod
    def is_bounding_box_contained_in_image_size(in_bounding_box, size_in_image):
        return (in_bounding_box[0][0] > 0 and in_bounding_box[0][1] < size_in_image[0] and
                in_bounding_box[1][0] > 0 and in_bounding_box[1][1] < size_in_image[1] and
                in_bounding_box[2][0] > 0 and in_bounding_box[2][1] < size_in_image[2])

    @staticmethod
    def fit_bounding_box_to_bounding_box(in_bounding_box, ref_bounding_box):
        return ((max(in_bounding_box[0][0], ref_bounding_box[0][0]), min(in_bounding_box[0][1], ref_bounding_box[0][1])),
                (max(in_bounding_box[1][0], ref_bounding_box[1][0]), min(in_bounding_box[1][1], ref_bounding_box[1][1])),
                (max(in_bounding_box[2][0], ref_bounding_box[2][0]), min(in_bounding_box[2][1], ref_bounding_box[2][1])))

    @staticmethod
    def fit_bounding_box_to_image(in_bounding_box, size_in_image):
        return ((max(in_bounding_box[0][0], 0), min(in_bounding_box[0][1], size_in_image[0])),
                (max(in_bounding_box[1][0], 0), min(in_bounding_box[1][1], size_in_image[1])),
                (max(in_bounding_box[2][0], 0), min(in_bounding_box[2][1], size_in_image[2])))

    @staticmethod
    def enlarge_bounding_box_to_bounding_box(in_bounding_box, ref_bounding_box):
        return ((min(in_bounding_box[0][0], ref_bounding_box[0][0]), max(in_bounding_box[0][1], ref_bounding_box[0][1])),
                (min(in_bounding_box[1][0], ref_bounding_box[1][0]), max(in_bounding_box[1][1], ref_bounding_box[1][1])),
                (min(in_bounding_box[2][0], ref_bounding_box[2][0]), max(in_bounding_box[2][1], ref_bounding_box[2][1])))

    @staticmethod
    def enlarge_bounding_box_to_image(in_bounding_box, size_in_image):
        return ((min(in_bounding_box[0][0], 0), max(in_bounding_box[0][1], size_in_image[0])),
                (min(in_bounding_box[1][0], 0), max(in_bounding_box[1][1], size_in_image[1])),
                (min(in_bounding_box[2][0], 0), max(in_bounding_box[2][1], size_in_image[2])))

    @staticmethod
    def translate_bounding_box(in_bounding_box, trans_dist):
        return ((in_bounding_box[0][0] + trans_dist[0], in_bounding_box[0][1] + trans_dist[0]),
                (in_bounding_box[1][0] + trans_dist[1], in_bounding_box[1][1] + trans_dist[1]),
                (in_bounding_box[2][0] + trans_dist[2], in_bounding_box[2][1] + trans_dist[2]))

    @staticmethod
    def dilate_bounding_box(in_bounding_box, size_borders):
        return ((in_bounding_box[0][0] - size_borders[0], in_bounding_box[0][1] + size_borders[0]),
                (in_bounding_box[1][0] - size_borders[1], in_bounding_box[1][1] + size_borders[1]),
                (in_bounding_box[2][0] - size_borders[2], in_bounding_box[2][1] + size_borders[2]))

    @staticmethod
    def erode_bounding_box(in_bounding_box, size_borders):
        return ((in_bounding_box[0][0] + size_borders[0], in_bounding_box[0][1] - size_borders[0]),
                (in_bounding_box[1][0] + size_borders[1], in_bounding_box[1][1] - size_borders[1]),
                (in_bounding_box[2][0] + size_borders[2], in_bounding_box[2][1] - size_borders[2]))


    @classmethod
    def compute_bounding_box_centered_bounding_box_fit_image(cls, in_bounding_box, out_size_bounding_box, size_in_image,
                                                             is_bounding_box_slices=False):
        if is_bounding_box_slices:
            out_size_bounding_box = (size_in_image[0], out_size_bounding_box[0], out_size_bounding_box[1])

        center_boundbox = cls.get_center_bounding_box(in_bounding_box)

        out_bounding_box = cls.get_create_bounding_box(center_boundbox, out_size_bounding_box)

        out_bounding_box = cls.translate_bounding_box_fit_image_size(out_bounding_box, size_in_image)

        return out_bounding_box

    @classmethod
    def compute_bounding_box_centered_image_fit_image(cls, out_size_bounding_box, size_in_image,
                                                      is_bounding_box_slices=False):
        in_image_bounding_box = cls.get_default_bounding_box_image(size_in_image)

        return cls.compute_bounding_box_centered_bounding_box_fit_image(in_image_bounding_box, out_size_bounding_box,
                                                                        size_in_image, is_bounding_box_slices)


    @classmethod
    def translate_bounding_box_fit_image_size(cls, in_bounding_box, size_in_image):
        size_in_bounding_box = cls.get_size_bounding_box(in_bounding_box)
        trans_dist = [0,0,0]
        for i in range(3):
            if size_in_bounding_box[i] < size_in_image[i]:
                trans_dist[i] = cls.get_translate_distance_fit_segment(in_bounding_box[i], size_in_image[i])
            else:
                trans_dist[i] = cls.get_translate_distance_fix_origin(in_bounding_box[i])
        #endfor
        if (trans_dist != [0,0,0]):
            return cls.translate_bounding_box(in_bounding_box, trans_dist)
        else:
            return in_bounding_box

    @staticmethod
    def get_translate_distance_fit_segment(bounding_limits, size_segment):
        if (bounding_limits[0] < 0):
            return -bounding_limits[0]
        elif (bounding_limits[1] > size_segment):
            return -(bounding_limits[1] - size_segment)
        else:
            return 0

    @staticmethod
    def get_translate_distance_fix_origin(bounding_limits):
        return -bounding_limits[0]


    @classmethod
    def compute_bounding_boxes_crop_extend_image(cls, in_bounding_box, size_in_image):

        out_crop_bounding_box = cls.fit_bounding_box_to_image(in_bounding_box, size_in_image)

        size_in_crop_bounding_box  = cls.get_size_bounding_box(in_bounding_box)
        size_out_crop_bounding_box = cls.get_size_bounding_box(out_crop_bounding_box)

        out_extend_bounding_box = cls.compute_bounding_box_centered_bounding_box_fit_image(in_bounding_box,
                                                                                           size_out_crop_bounding_box,
                                                                                           size_in_crop_bounding_box)
        return (out_crop_bounding_box, out_extend_bounding_box)


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
        out_bounding_box = cls.dilate_bounding_box(out_bounding_box, size_borders_buffer)
        out_bounding_box = cls.fit_bounding_box_to_image(out_bounding_box, in_mask_array.shape)

        if is_bounding_box_slices:
            return ((0, in_mask_array.shape[0]),
                    (out_bounding_box[1][0], out_bounding_box[1][1]),
                    (out_bounding_box[2][0], out_bounding_box[2][1]))
        else:
            return out_bounding_box