
from typing import Tuple, Union
import numpy as np

from common.exceptionmanager import catch_error_exception
from imageoperators.boundingboxes import BoundBox3DType, BoundBox2DType
from imageoperators.imageoperator import CropImage
from preprocessing.imagegenerator import ImageGenerator


class RandomWindowImages(ImageGenerator):

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 num_images: int,
                 size_volume_image: Union[Tuple[int, int, int], Tuple[int, int]] = (0, 0, 0)
                 ) -> None:
        super(RandomWindowImages, self).__init__(size_image, num_images)

        self._ndims = len(self._size_image)
        if np.isscalar(size_volume_image):
            self._size_volume_image = tuple([size_volume_image] * self._ndims)
        else:
            self._size_volume_image = size_volume_image

        if self._ndims == 2:
            self._func_crop_images = CropImage._compute2d
        elif self._ndims == 3:
            self._func_crop_images = CropImage._compute3d
        else:
            message = 'RandomWindowImages:__init__: wrong \'ndims\': %s...' % (self._ndims)
            catch_error_exception(message)

    def update_image_data(self, in_shape_image: Tuple[int, ...], seed_0: int = None) -> None:
        self._size_volume_image = in_shape_image[0:self._ndims]

    def _compute_gendata(self, **kwargs) -> None:
        seed = kwargs['seed']
        self._crop_window_boundbox = self._get_random_crop_boundbox_image(seed)
        self._is_compute_gendata = False

    def _initialize_gendata(self) -> None:
        self._is_compute_gendata = True
        self._crop_window_boundbox = None

    def _get_image(self, in_image: np.ndarray) -> np.ndarray:
        return self._func_crop_images(in_image, self._crop_window_boundbox)

    def _get_random_origin_crop_windowbox_image(self, seed: int = None) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        if seed is not None:
            np.random.seed(seed)

        origin_crop_boundbox = []
        for i in range(self._ndims):
            searching_space_1d = self._size_volume_image[i] - self._size_image[i]
            origin_1d = np.random.randint(searching_space_1d + 1)
            origin_crop_boundbox.append(origin_1d)

        if self._ndims == 3:
            return (origin_crop_boundbox[0], origin_crop_boundbox[1], origin_crop_boundbox[2])
        else:
            return (origin_crop_boundbox[0], origin_crop_boundbox[1])

    def _get_random_crop_boundbox_image(self, seed: int = None) -> Union[BoundBox3DType, BoundBox2DType]:
        origin_crop_boundbox = self._get_random_origin_crop_windowbox_image(seed=seed)

        crop_boundbox = []
        for i in range(self._ndims):
            limit_left = origin_crop_boundbox[i]
            limit_right = origin_crop_boundbox[i] + self._size_image[i]
            crop_boundbox.append((limit_left, limit_right))

        if self._ndims == 3:
            return (crop_boundbox[0], crop_boundbox[1], crop_boundbox[2])
        else:
            return (crop_boundbox[0], crop_boundbox[1])

    def get_cropped_image(self, in_image: np.ndarray, seed: int = None) -> np.ndarray:
        crop_boundbox = self._get_random_crop_boundbox_image(seed=seed)
        return self._func_crop_images(in_image, crop_boundbox)

    def get_text_description(self) -> str:
        message = 'Random-window generation of images:\n'
        message += 'size image: \'%s\', size volume image: \'%s\'. Num random patches per volume: \'%s\'...\n' \
                   % (str(self._size_image), str(self._size_volume_image), self._num_images)
        return message
