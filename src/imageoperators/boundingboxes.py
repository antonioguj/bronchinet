
from typing import Tuple, Union
import numpy as np

BoundBox3DType = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
BoundBox2DType = Tuple[Tuple[int, int], Tuple[int, int]]


class BoundingBoxes(object):

    @staticmethod
    def get_size_boundbox(in_boundbox: BoundBox3DType) -> Tuple[int, int, int]:
        return ((in_boundbox[0][1] - in_boundbox[0][0]),
                (in_boundbox[1][1] - in_boundbox[1][0]),
                (in_boundbox[2][1] - in_boundbox[2][0]))

    @staticmethod
    def get_max_size_boundbox(size_boundbox: Tuple[int, int, int],
                              max_size_boundbox: Tuple[int, int, int]
                              ) -> Tuple[int, int, int]:
        return (max(size_boundbox[0], max_size_boundbox[0]),
                max(size_boundbox[1], max_size_boundbox[1]),
                max(size_boundbox[2], max_size_boundbox[2]))

    @staticmethod
    def get_min_size_boundbox(size_boundbox: Tuple[int, int, int],
                              min_size_boundbox: Tuple[int, int, int]
                              ) -> Tuple[int, int, int]:
        return (min(size_boundbox[0], min_size_boundbox[0]),
                min(size_boundbox[1], min_size_boundbox[1]),
                min(size_boundbox[2], min_size_boundbox[2]))

    @staticmethod
    def get_coords0_boundbox(in_boundbox: BoundBox3DType) -> Tuple[int, int, int]:
        return (in_boundbox[0][0], in_boundbox[1][0], in_boundbox[2][0])

    @staticmethod
    def get_center_boundbox(in_boundbox: BoundBox3DType) -> Tuple[int, int, int]:
        return (int((in_boundbox[0][0] + in_boundbox[0][1]) / 2),
                int((in_boundbox[1][0] + in_boundbox[1][1]) / 2),
                int((in_boundbox[2][0] + in_boundbox[2][1]) / 2))

    @staticmethod
    def get_create_boundbox(center_boundbox: Tuple[int, int, int],
                            size_boundbox: Tuple[int, int, int]
                            ) -> BoundBox3DType:
        begin_boundbox = (center_boundbox[0] - int(size_boundbox[0] / 2),
                          center_boundbox[1] - int(size_boundbox[1] / 2),
                          center_boundbox[2] - int(size_boundbox[2] / 2))
        return ((begin_boundbox[0], begin_boundbox[0] + size_boundbox[0]),
                (begin_boundbox[1], begin_boundbox[1] + size_boundbox[1]),
                (begin_boundbox[2], begin_boundbox[2] + size_boundbox[2]))

    @staticmethod
    def get_default_boundbox_image(size_image: Tuple[int, int, int]) -> BoundBox3DType:
        return ((0, size_image[0]), (0, size_image[1]), (0, size_image[2]))

    @staticmethod
    def is_boundbox_inside_boundbox(in_boundbox: BoundBox3DType,
                                    ref_boundbox: BoundBox3DType
                                    ) -> bool:
        return (in_boundbox[0][0] >= ref_boundbox[0][0] and in_boundbox[0][1] <= ref_boundbox[0][1]
                and in_boundbox[1][0] >= ref_boundbox[1][0] and in_boundbox[1][1] <= ref_boundbox[1][1]
                and in_boundbox[2][0] >= ref_boundbox[2][0] and in_boundbox[2][1] <= ref_boundbox[2][1])

    @staticmethod
    def is_boundbox_inside_image_size(in_boundbox: BoundBox3DType,
                                      size_image: Tuple[int, int, int]
                                      ) -> bool:
        return (in_boundbox[0][0] >= 0 and in_boundbox[0][1] <= size_image[0]
                and in_boundbox[1][0] >= 0 and in_boundbox[1][1] <= size_image[1]
                and in_boundbox[2][0] >= 0 and in_boundbox[2][1] <= size_image[2])

    @staticmethod
    def is_image_inside_boundbox(size_in_boundbox: Tuple[int, int, int],
                                 size_image: Tuple[int, int, int]
                                 ) -> bool:
        return (size_image[0] <= size_in_boundbox[0]
                and size_image[1] <= size_in_boundbox[1]
                and size_image[2] <= size_in_boundbox[2])

    @staticmethod
    def fit_boundbox_to_boundbox(in_boundbox: BoundBox3DType,
                                 ref_boundbox: BoundBox3DType
                                 ) -> BoundBox3DType:
        return ((max(in_boundbox[0][0], ref_boundbox[0][0]), min(in_boundbox[0][1], ref_boundbox[0][1])),
                (max(in_boundbox[1][0], ref_boundbox[1][0]), min(in_boundbox[1][1], ref_boundbox[1][1])),
                (max(in_boundbox[2][0], ref_boundbox[2][0]), min(in_boundbox[2][1], ref_boundbox[2][1])))

    @staticmethod
    def fit_boundbox_to_image(in_boundbox: BoundBox3DType,
                              size_image: Tuple[int, int, int]
                              ) -> BoundBox3DType:
        return ((max(in_boundbox[0][0], 0), min(in_boundbox[0][1], size_image[0])),
                (max(in_boundbox[1][0], 0), min(in_boundbox[1][1], size_image[1])),
                (max(in_boundbox[2][0], 0), min(in_boundbox[2][1], size_image[2])))

    @staticmethod
    def enlarge_boundbox_to_boundbox(in_boundbox: BoundBox3DType,
                                     ref_boundbox: BoundBox3DType
                                     ) -> BoundBox3DType:
        return ((min(in_boundbox[0][0], ref_boundbox[0][0]), max(in_boundbox[0][1], ref_boundbox[0][1])),
                (min(in_boundbox[1][0], ref_boundbox[1][0]), max(in_boundbox[1][1], ref_boundbox[1][1])),
                (min(in_boundbox[2][0], ref_boundbox[2][0]), max(in_boundbox[2][1], ref_boundbox[2][1])))

    @staticmethod
    def enlarge_boundbox_to_image(in_boundbox: BoundBox3DType,
                                  size_image: Tuple[int, int, int]
                                  ) -> BoundBox3DType:
        return ((min(in_boundbox[0][0], 0), max(in_boundbox[0][1], size_image[0])),
                (min(in_boundbox[1][0], 0), max(in_boundbox[1][1], size_image[1])),
                (min(in_boundbox[2][0], 0), max(in_boundbox[2][1], size_image[2])))

    @staticmethod
    def translate_boundbox(in_boundbox: BoundBox3DType,
                           trans_dist: Tuple[int, int, int]
                           ) -> BoundBox3DType:
        return ((in_boundbox[0][0] + trans_dist[0], in_boundbox[0][1] + trans_dist[0]),
                (in_boundbox[1][0] + trans_dist[1], in_boundbox[1][1] + trans_dist[1]),
                (in_boundbox[2][0] + trans_dist[2], in_boundbox[2][1] + trans_dist[2]))

    @staticmethod
    def dilate_boundbox(in_boundbox: BoundBox3DType,
                        size_borders: Tuple[int, int, int]
                        ) -> BoundBox3DType:
        return ((in_boundbox[0][0] - size_borders[0], in_boundbox[0][1] + size_borders[0]),
                (in_boundbox[1][0] - size_borders[1], in_boundbox[1][1] + size_borders[1]),
                (in_boundbox[2][0] - size_borders[2], in_boundbox[2][1] + size_borders[2]))

    @staticmethod
    def erode_boundbox(in_boundbox: BoundBox3DType,
                       size_borders: Tuple[int, int, int]
                       ) -> BoundBox3DType:
        return ((in_boundbox[0][0] + size_borders[0], in_boundbox[0][1] - size_borders[0]),
                (in_boundbox[1][0] + size_borders[1], in_boundbox[1][1] - size_borders[1]),
                (in_boundbox[2][0] + size_borders[2], in_boundbox[2][1] - size_borders[2]))

    @classmethod
    def calc_boundbox_centered_boundbox_fitimg(cls, in_boundbox: BoundBox3DType,
                                               out_size_boundbox: Tuple[int, int, int],
                                               size_image: Tuple[int, int, int]
                                               ) -> BoundBox3DType:
        center_boundbox = cls.get_center_boundbox(in_boundbox)
        out_boundbox = cls.get_create_boundbox(center_boundbox, out_size_boundbox)
        out_boundbox = cls.translate_boundbox_fitimg(out_boundbox, size_image)
        return out_boundbox

    @classmethod
    def calc_boundbox_centered_image_fitimg(cls, out_size_boundbox: Tuple[int, int, int],
                                            size_image: Tuple[int, int, int]
                                            ) -> BoundBox3DType:
        in_image_boundbox = cls.get_default_boundbox_image(size_image)
        out_boundbox = cls.calc_boundbox_centered_boundbox_fitimg(in_image_boundbox, out_size_boundbox, size_image)
        return out_boundbox

    @classmethod
    def translate_boundbox_fitimg(cls, in_boundbox: BoundBox3DType,
                                  size_image: Tuple[int, int, int]
                                  ) -> BoundBox3DType:
        size_in_boundbox = cls.get_size_boundbox(in_boundbox)
        trans_dist = [0, 0, 0]
        for i in range(3):
            if size_in_boundbox[i] < size_image[i]:
                trans_dist[i] = cls.get_translate_distance_fitseg(in_boundbox[i], size_image[i])
            else:
                trans_dist[i] = cls.get_translate_distance_fixorigin(in_boundbox[i])

        trans_dist = (trans_dist[0], trans_dist[1], trans_dist[2])
        if trans_dist != (0, 0, 0):
            return cls.translate_boundbox(in_boundbox, trans_dist)
        else:
            return in_boundbox

    @staticmethod
    def get_translate_distance_fitseg(bound_limits: Tuple[int, int], size_segment: int) -> int:
        if bound_limits[0] < 0:
            return -bound_limits[0]
        elif bound_limits[1] > size_segment:
            return -(bound_limits[1] - size_segment)
        else:
            return 0

    @staticmethod
    def get_translate_distance_fixorigin(bound_limits: Tuple[int, int]) -> int:
        return -bound_limits[0]

    @classmethod
    def calc_boundboxes_crop_extend_image(cls, in_boundbox: BoundBox3DType,
                                          size_image: Tuple[int, int, int]
                                          ) -> Tuple[BoundBox3DType, BoundBox3DType]:
        out_crop_boundbox = cls.fit_boundbox_to_image(in_boundbox, size_image)
        size_in_crop_boundbox = cls.get_size_boundbox(in_boundbox)
        size_out_crop_boundbox = cls.get_size_boundbox(out_crop_boundbox)
        out_extend_boundbox = cls.calc_boundbox_centered_boundbox_fitimg(in_boundbox, size_out_crop_boundbox,
                                                                         size_in_crop_boundbox)
        return (out_crop_boundbox, out_extend_boundbox)

    @classmethod
    def calc_boundboxes_crop_extend_image_reverse(cls, in_boundbox: BoundBox3DType,
                                                  size_image: Tuple[int, int, int]
                                                  ) -> Tuple[BoundBox3DType, BoundBox3DType]:
        (out_extend_boundbox, out_crop_boundbox) = cls.calc_boundboxes_crop_extend_image(in_boundbox, size_image)
        return (out_crop_boundbox, out_extend_boundbox)

    @staticmethod
    def calc_split_boundboxes(in_boundbox: BoundBox3DType, axis: int = 0
                              ) -> Union[Tuple[BoundBox3DType, BoundBox3DType], None]:
        if axis == 0:
            half_boundbox_zdim = int((in_boundbox[0][1] + in_boundbox[0][0]) / 2)
            out_boundbox_1 = ((in_boundbox[0][0], half_boundbox_zdim),
                              (in_boundbox[1][0], in_boundbox[1][1]),
                              (in_boundbox[2][0], in_boundbox[2][1]))
            out_boundbox_2 = ((half_boundbox_zdim, in_boundbox[0][1]),
                              (in_boundbox[1][0], in_boundbox[1][1]),
                              (in_boundbox[2][0], in_boundbox[2][1]))
            return (out_boundbox_1, out_boundbox_2)
        elif axis == 1:
            half_boundbox_xdim = int((in_boundbox[1][1] + in_boundbox[1][0]) / 2)
            out_boundbox_1 = ((in_boundbox[0][0], in_boundbox[0][1]),
                              (in_boundbox[1][0], half_boundbox_xdim),
                              (in_boundbox[2][0], in_boundbox[2][1]))
            out_boundbox_2 = ((in_boundbox[0][0], in_boundbox[0][1]),
                              (half_boundbox_xdim, in_boundbox[1][1]),
                              (in_boundbox[2][0], in_boundbox[2][1]))
            return (out_boundbox_1, out_boundbox_2)
        elif axis == 2:
            half_boundbox_ydim = int((in_boundbox[2][1] + in_boundbox[2][0]) / 2)
            out_boundbox_1 = ((in_boundbox[0][0], in_boundbox[0][1]),
                              (in_boundbox[1][0], in_boundbox[1][1]),
                              (in_boundbox[2][0], half_boundbox_ydim))
            out_boundbox_2 = ((in_boundbox[0][0], in_boundbox[0][1]),
                              (in_boundbox[1][0], in_boundbox[1][1]),
                              (half_boundbox_ydim, in_boundbox[2][1]))
            return (out_boundbox_1, out_boundbox_2)
        else:
            return None

    @classmethod
    def compute_boundbox_contain_mask(cls, in_mask: np.ndarray,
                                      size_borders_buffer: Tuple[int, int, int] = (0, 0, 0),
                                      is_boundbox_slices: bool = False
                                      ) -> BoundBox3DType:
        # find where there are active masks. Fit bounding box that contain all masks
        indexes_posit_mask = np.argwhere(in_mask != 0)
        out_boundbox = ((min(indexes_posit_mask[:, 0]), max(indexes_posit_mask[:, 0])),
                        (min(indexes_posit_mask[:, 1]), max(indexes_posit_mask[:, 1])),
                        (min(indexes_posit_mask[:, 2]), max(indexes_posit_mask[:, 2])))

        # enlarge bounding box to account for border effects
        out_boundbox = cls.dilate_boundbox(out_boundbox, size_borders_buffer)
        out_boundbox = cls.fit_boundbox_to_image(out_boundbox, in_mask.shape)

        if is_boundbox_slices:
            return ((0, in_mask.shape[0]),
                    (out_boundbox[1][0], out_boundbox[1][1]),
                    (out_boundbox[2][0], out_boundbox[2][1]))
        else:
            return out_boundbox
