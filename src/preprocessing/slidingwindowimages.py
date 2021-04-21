
from typing import Tuple, List, Union
import numpy as np

from common.exceptionmanager import catch_error_exception
from imageoperators.boundingboxes import BoundBox3DType, BoundBox2DType
from imageoperators.imageoperator import CropImage
from preprocessing.imagegenerator import ImageGenerator


class SlidingWindowImages(ImageGenerator):

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 prop_overlap_images: Union[Tuple[float, float, float], Tuple[float, float]],
                 size_volume_image: Union[Tuple[int, int, int], Tuple[int, int]] = (0, 0, 0)
                 ) -> None:
        super(SlidingWindowImages, self).__init__(size_image, num_images=1)

        self._ndims = len(size_image)
        self._prop_overlap_images = prop_overlap_images
        self._size_volume_image = size_volume_image

        self._num_images_dirs = self._get_num_images_dirs()
        self._num_images = np.prod(self._num_images_dirs)

        if self._ndims == 2:
            self._func_get_indexes_local = self.get_indexes_local_2dim
            self._func_crop_images = CropImage._compute2d

        elif self._ndims == 3:
            self._func_get_indexes_local = self.get_indexes_local_3dim
            self._func_crop_images = CropImage._compute3d
        else:
            message = 'SlidingWindowImages:__init__: wrong \'ndims\': %s' % (self._ndims)
            catch_error_exception(message)

        self._initialize_gendata()

    @staticmethod
    def get_num_images_1d(size_segment: int,
                          prop_overlap_segment: float,
                          size_total_segment: int
                          ) -> int:
        return max(int(np.ceil((size_total_segment - prop_overlap_segment * size_segment)
                               / (1 - prop_overlap_segment) / size_segment)), 0)

    @staticmethod
    def get_limits_image_1d(index: int,
                            size_segment: int,
                            prop_overlap_segment: float,
                            size_total_segment: int
                            ) -> Tuple[int, int]:
        coord_n = int(index * (1.0 - prop_overlap_segment) * size_segment)
        coord_npl1 = coord_n + size_segment
        if coord_npl1 > size_total_segment:
            coord_npl1 = size_total_segment
            coord_n = size_total_segment - size_segment
        return (coord_n, coord_npl1)

    @staticmethod
    def get_indexes_local_2dim(index: int,
                               num_images_dirs: Tuple[int, int]
                               ) -> Tuple[int, int]:
        num_images_x = num_images_dirs[0]
        index_y = index // num_images_x
        index_x = index % num_images_x
        return (index_x, index_y)

    @staticmethod
    def get_indexes_local_3dim(index: int,
                               num_images_dirs: Tuple[int, int, int]
                               ) -> Tuple[int, int, int]:
        num_images_x = num_images_dirs[1]
        num_images_y = num_images_dirs[2]
        num_images_xy = num_images_x * num_images_y
        index_z = index // (num_images_xy)
        index_xy = index % (num_images_xy)
        index_y = index_xy // num_images_x
        index_x = index_xy % num_images_x
        return (index_z, index_x, index_y)

    def update_image_data(self, in_shape_image: Tuple[int, ...]) -> None:
        self._size_volume_image = in_shape_image[0:self._ndims]
        self._num_images_dirs = self._get_num_images_dirs()
        self._num_images = np.prod(self._num_images_dirs)

    def _initialize_gendata(self) -> None:
        self._crop_boundbox = None

    def _update_gendata(self, **kwargs) -> None:
        index = kwargs['index']
        self._crop_boundbox = self._get_crop_boundbox_image(index)

    def _get_image(self, in_image: np.ndarray) -> np.ndarray:
        return self._func_crop_images(in_image, self._crop_boundbox)

    def get_cropped_image(self, in_image: np.ndarray, index: int) -> np.ndarray:
        crop_boundbox = self._get_crop_boundbox_image(index)
        return self._func_crop_images(in_image, crop_boundbox)

    def _get_crop_boundbox_image(self, index: int) -> Union[BoundBox3DType, BoundBox2DType]:
        indexes_local = self._func_get_indexes_local(index, self._num_images_dirs)
        crop_boundbox = []
        for i in range(self._ndims):
            (limit_left, limit_right) = self.get_limits_image_1d(indexes_local[i], self._size_image[i],
                                                                 self._prop_overlap_images[i],
                                                                 self._size_volume_image[i])
            crop_boundbox.append((limit_left, limit_right))

        if self._ndims == 3:
            return (crop_boundbox[0], crop_boundbox[1], crop_boundbox[2])
        else:
            return (crop_boundbox[0], crop_boundbox[1])

    def _get_num_images_dirs(self) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        num_images_dirs = []
        for i in range(self._ndims):
            num_images_1d = self.get_num_images_1d(self._size_image[i],
                                                   self._prop_overlap_images[i],
                                                   self._size_volume_image[i])
            num_images_dirs.append(num_images_1d)

        if self._ndims == 3:
            return (num_images_dirs[0], num_images_dirs[1], num_images_dirs[2])
        else:
            return (num_images_dirs[0], num_images_dirs[1])

    def get_limits_sliding_window_image(self) -> List[List[Tuple[int, int]]]:
        limits_window_image = []
        for i in range(self._ndims):
            limits_image_1dir = \
                [self.get_limits_image_1d(index, self._size_image[i],
                                          self._prop_overlap_images[i],
                                          self._size_volume_image[i]) for index in range(self._num_images_dirs[i])]
            limits_window_image.append(limits_image_1dir)

        return limits_window_image

    def get_text_description(self) -> str:
        message = 'Sliding-window generation of image patches:\n'
        message += '- size image: \'%s\', prop. overlap: \'%s\', size volume: \'%s\'...\n' \
                   % (str(self._size_image), str(self._prop_overlap_images), str(self._size_volume_image))
        message += '- num images total: \'%s\', and num images in each direction: \'%s\'...\n' \
                   % (self._num_images, str(self._num_images_dirs))
        limits_window_image = self.get_limits_sliding_window_image()
        for i in range(self._ndims):
            message += '- limits bound-boxes in dir \'%s\': \'%s\'...\n' % (i, str(limits_window_image[i]))

        return message


class SlicingImages(SlidingWindowImages):
    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 size_volume_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 ) -> None:
        super(SlicingImages, self).__init__(size_image,
                                            prop_overlap_images=(0.0, 0.0, 0.0),
                                            size_volume_image=size_volume_image)

    @staticmethod
    def get_num_images_1d(size_segment: int,
                          prop_overlap_segment: float,
                          size_total_segment: int
                          ) -> int:
        return max(int(np.floor((size_total_segment - prop_overlap_segment * size_segment)
                                / (1 - prop_overlap_segment) / size_segment)), 0)
