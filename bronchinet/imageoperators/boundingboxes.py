
from typing import Tuple
import numpy as np

BoundBox3DType = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


class BoundingBoxes(object):

    @staticmethod
    def get_size_bounding_box(bounding_box: BoundBox3DType) -> Tuple[int, int, int]:
        return ((bounding_box[0][1] - bounding_box[0][0]),
                (bounding_box[1][1] - bounding_box[1][0]),
                (bounding_box[2][1] - bounding_box[2][0]))

    @staticmethod
    def get_max_size_bounding_box(size_bounding_box: Tuple[int, int, int],
                                  max_size_bounding_box: Tuple[int, int, int]
                                  ) -> Tuple[int, int, int]:
        return (max(size_bounding_box[0], max_size_bounding_box[0]),
                max(size_bounding_box[1], max_size_bounding_box[1]),
                max(size_bounding_box[2], max_size_bounding_box[2]))

    @staticmethod
    def get_min_size_bounding_box(size_bounding_box: Tuple[int, int, int],
                                  min_size_bounding_box: Tuple[int, int, int]
                                  ) -> Tuple[int, int, int]:
        return (min(size_bounding_box[0], min_size_bounding_box[0]),
                min(size_bounding_box[1], min_size_bounding_box[1]),
                min(size_bounding_box[2], min_size_bounding_box[2]))

    @staticmethod
    def get_coords0_bounding_box(bounding_box: BoundBox3DType) -> Tuple[int, int, int]:
        return (bounding_box[0][0], bounding_box[1][0], bounding_box[2][0])

    @staticmethod
    def get_center_bounding_box(bounding_box: BoundBox3DType) -> Tuple[int, int, int]:
        return (int( (bounding_box[0][0] + bounding_box[0][1]) / 2),
                int( (bounding_box[1][0] + bounding_box[1][1]) / 2),
                int( (bounding_box[2][0] + bounding_box[2][1]) / 2))

    @staticmethod
    def get_create_bounding_box(center_boundbox: Tuple[int, int, int],
                                size_bounding_box: Tuple[int, int, int]
                                ) -> BoundBox3DType:
        begin_bounding_box = (center_boundbox[0] - int( size_bounding_box[0] / 2),
                              center_boundbox[1] - int( size_bounding_box[1] / 2),
                              center_boundbox[2] - int( size_bounding_box[2] / 2))
        return ((begin_bounding_box[0], begin_bounding_box[0] + size_bounding_box[0]),
                (begin_bounding_box[1], begin_bounding_box[1] + size_bounding_box[1]),
                (begin_bounding_box[2], begin_bounding_box[2] + size_bounding_box[2]))

    @staticmethod
    def get_default_bounding_box_image(size_image: Tuple[int, int, int]) -> BoundBox3DType:
        return ((0, size_image[0]), (0, size_image[1]), (0, size_image[2]))

    @staticmethod
    def is_bounding_box_contained_in_bounding_box(in_bounding_box: BoundBox3DType,
                                                  ref_bounding_box: BoundBox3DType
                                                  ) -> bool:
        return (in_bounding_box[0][0] >= ref_bounding_box[0][0] and in_bounding_box[0][1] <= ref_bounding_box[0][1] and
                in_bounding_box[1][0] >= ref_bounding_box[1][0] and in_bounding_box[1][1] <= ref_bounding_box[1][1] and
                in_bounding_box[2][0] >= ref_bounding_box[2][0] and in_bounding_box[2][1] <= ref_bounding_box[2][1])

    @staticmethod
    def is_bounding_box_contained_in_image_size(in_bounding_box: BoundBox3DType,
                                                size_image: Tuple[int, int, int]
                                                ) -> bool:
        return (in_bounding_box[0][0] >= 0 and in_bounding_box[0][1] <= size_image[0] and
                in_bounding_box[1][0] >= 0 and in_bounding_box[1][1] <= size_image[1] and
                in_bounding_box[2][0] >= 0 and in_bounding_box[2][1] <= size_image[2])

    @staticmethod
    def is_image_patch_contained_in_bounding_box(size_in_bounding_box: Tuple[int, int, int],
                                                 size_image: Tuple[int, int, int]
                                                 ) -> bool:
        return (size_image[0] <= size_in_bounding_box[0] and
                size_image[1] <= size_in_bounding_box[1] and
                size_image[2] <= size_in_bounding_box[2])

    @staticmethod
    def fit_bounding_box_to_bounding_box(in_bounding_box: BoundBox3DType,
                                         ref_bounding_box: BoundBox3DType
                                         ) -> BoundBox3DType:
        return ((max(in_bounding_box[0][0], ref_bounding_box[0][0]), min(in_bounding_box[0][1], ref_bounding_box[0][1])),
                (max(in_bounding_box[1][0], ref_bounding_box[1][0]), min(in_bounding_box[1][1], ref_bounding_box[1][1])),
                (max(in_bounding_box[2][0], ref_bounding_box[2][0]), min(in_bounding_box[2][1], ref_bounding_box[2][1])))

    @staticmethod
    def fit_bounding_box_to_image(in_bounding_box: BoundBox3DType,
                                  size_image: Tuple[int, int, int]
                                  ) -> BoundBox3DType:
        return ((max(in_bounding_box[0][0], 0), min(in_bounding_box[0][1], size_image[0])),
                (max(in_bounding_box[1][0], 0), min(in_bounding_box[1][1], size_image[1])),
                (max(in_bounding_box[2][0], 0), min(in_bounding_box[2][1], size_image[2])))

    @staticmethod
    def enlarge_bounding_box_to_bounding_box(in_bounding_box: BoundBox3DType,
                                             ref_bounding_box: BoundBox3DType
                                             ) -> BoundBox3DType:
        return ((min(in_bounding_box[0][0], ref_bounding_box[0][0]), max(in_bounding_box[0][1], ref_bounding_box[0][1])),
                (min(in_bounding_box[1][0], ref_bounding_box[1][0]), max(in_bounding_box[1][1], ref_bounding_box[1][1])),
                (min(in_bounding_box[2][0], ref_bounding_box[2][0]), max(in_bounding_box[2][1], ref_bounding_box[2][1])))

    @staticmethod
    def enlarge_bounding_box_to_image(in_bounding_box: BoundBox3DType,
                                      size_image: Tuple[int, int, int]
                                      ) -> BoundBox3DType:
        return ((min(in_bounding_box[0][0], 0), max(in_bounding_box[0][1], size_image[0])),
                (min(in_bounding_box[1][0], 0), max(in_bounding_box[1][1], size_image[1])),
                (min(in_bounding_box[2][0], 0), max(in_bounding_box[2][1], size_image[2])))

    @staticmethod
    def translate_bounding_box(in_bounding_box: BoundBox3DType,
                               trans_dist: Tuple[int, int, int]
                               ) -> BoundBox3DType:
        return ((in_bounding_box[0][0] + trans_dist[0], in_bounding_box[0][1] + trans_dist[0]),
                (in_bounding_box[1][0] + trans_dist[1], in_bounding_box[1][1] + trans_dist[1]),
                (in_bounding_box[2][0] + trans_dist[2], in_bounding_box[2][1] + trans_dist[2]))

    @staticmethod
    def dilate_bounding_box(in_bounding_box: BoundBox3DType,
                            size_borders: Tuple[int, int, int]
                            ) -> BoundBox3DType:
        return ((in_bounding_box[0][0] - size_borders[0], in_bounding_box[0][1] + size_borders[0]),
                (in_bounding_box[1][0] - size_borders[1], in_bounding_box[1][1] + size_borders[1]),
                (in_bounding_box[2][0] - size_borders[2], in_bounding_box[2][1] + size_borders[2]))

    @staticmethod
    def erode_bounding_box(in_bounding_box: BoundBox3DType,
                           size_borders: Tuple[int, int, int]
                           ) -> BoundBox3DType:
        return ((in_bounding_box[0][0] + size_borders[0], in_bounding_box[0][1] - size_borders[0]),
                (in_bounding_box[1][0] + size_borders[1], in_bounding_box[1][1] - size_borders[1]),
                (in_bounding_box[2][0] + size_borders[2], in_bounding_box[2][1] - size_borders[2]))

    @classmethod
    def compute_bounding_box_centered_bounding_box_fit_image(cls,
                                                             in_bounding_box: BoundBox3DType,
                                                             out_size_bounding_box: Tuple[int, int, int],
                                                             size_image: Tuple[int, int, int]
                                                             ) -> BoundBox3DType:
        center_boundbox = cls.get_center_bounding_box(in_bounding_box)
        out_bounding_box = cls.get_create_bounding_box(center_boundbox,
                                                       out_size_bounding_box)
        out_bounding_box = cls.translate_bounding_box_fit_image_size(out_bounding_box,
                                                                     size_image)
        return out_bounding_box

    @classmethod
    def compute_bounding_box_centered_image_fit_image(cls,
                                                      out_size_bounding_box: Tuple[int, int, int],
                                                      size_image: Tuple[int, int, int]
                                                      ) -> BoundBox3DType:
        in_image_bounding_box = cls.get_default_bounding_box_image(size_image)
        out_bounding_box = cls.compute_bounding_box_centered_bounding_box_fit_image(in_image_bounding_box,
                                                                                    out_size_bounding_box,
                                                                                    size_image)
        return out_bounding_box

    @classmethod
    def translate_bounding_box_fit_image_size(cls,
                                              in_bounding_box: BoundBox3DType,
                                              size_image: Tuple[int, int, int]
                                              ) -> BoundBox3DType:
        size_in_bounding_box = cls.get_size_bounding_box(in_bounding_box)
        trans_dist = [0,0,0]
        for i in range(3):
            if size_in_bounding_box[i] < size_image[i]:
                trans_dist[i] = cls.get_translate_distance_fit_segment(in_bounding_box[i], size_image[i])
            else:
                trans_dist[i] = cls.get_translate_distance_fix_origin(in_bounding_box[i])

        if (trans_dist != [0,0,0]):
            return cls.translate_bounding_box(in_bounding_box, trans_dist)
        else:
            return in_bounding_box

    @staticmethod
    def get_translate_distance_fit_segment(bounding_limits: Tuple[int, int],
                                           size_segment: int
                                           ) -> int:
        if (bounding_limits[0] < 0):
            return -bounding_limits[0]
        elif (bounding_limits[1] > size_segment):
            return -(bounding_limits[1] - size_segment)
        else:
            return 0

    @staticmethod
    def get_translate_distance_fix_origin(bounding_limits: Tuple[int, int]) -> int:
        return -bounding_limits[0]

    @classmethod
    def compute_bounding_boxes_crop_extend_image(cls,
                                                 in_bounding_box: BoundBox3DType,
                                                 size_image: Tuple[int, int, int]
                                                 ) -> Tuple[BoundBox3DType, BoundBox3DType]:
        out_crop_bounding_box = cls.fit_bounding_box_to_image(in_bounding_box,
                                                              size_image)
        size_in_crop_bounding_box  = cls.get_size_bounding_box(in_bounding_box)
        size_out_crop_bounding_box = cls.get_size_bounding_box(out_crop_bounding_box)

        out_extend_bounding_box = cls.compute_bounding_box_centered_bounding_box_fit_image(in_bounding_box,
                                                                                           size_out_crop_bounding_box,
                                                                                           size_in_crop_bounding_box)
        return (out_crop_bounding_box, out_extend_bounding_box)

    @classmethod
    def compute_bounding_boxes_crop_extend_image_reverse(cls,
                                                         in_bounding_box: BoundBox3DType,
                                                         size_image: Tuple[int, int, int]
                                                         ) -> Tuple[BoundBox3DType, BoundBox3DType]:
        (out_extend_bounding_box, out_crop_bounding_box) = cls.compute_bounding_boxes_crop_extend_image(in_bounding_box,
                                                                                                        size_image)
        return (out_crop_bounding_box, out_extend_bounding_box)

    @staticmethod
    def compute_split_bounding_boxes(in_bounding_box: BoundBox3DType, axis: int = 0) -> Tuple[BoundBox3DType, BoundBox3DType]:
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
    def compute_bounding_box_contain_masks(cls,
                                           in_mask: np.ndarray,
                                           size_borders_buffer: Tuple[int, int, int] = (0, 0, 0),
                                           is_bounding_box_slices: bool = False
                                           ) -> BoundBox3DType:
        # find where there are active masks. Fit bounding box that contain all masks
        indexes_active_masks = np.argwhere(in_mask != 0)
        out_bounding_box = ((min(indexes_active_masks[:,0]), max(indexes_active_masks[:,0])),
                            (min(indexes_active_masks[:,1]), max(indexes_active_masks[:,1])),
                            (min(indexes_active_masks[:,2]), max(indexes_active_masks[:,2])))

        # enlarge bounding box to account for border effects
        out_bounding_box = cls.dilate_bounding_box(out_bounding_box, size_borders_buffer)
        out_bounding_box = cls.fit_bounding_box_to_image(out_bounding_box, in_mask.shape)

        if is_bounding_box_slices:
            return ((0, in_mask.shape[0]),
                    (out_bounding_box[1][0], out_bounding_box[1][1]),
                    (out_bounding_box[2][0], out_bounding_box[2][1]))
        else:
            return out_bounding_box