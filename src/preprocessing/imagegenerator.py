
from typing import Tuple, List, Union
import numpy as np

from common.functionutil import ImagesUtil


class ImageGenerator(object):

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 num_images: int
                 ) -> None:
        self._size_image = size_image
        self._num_images = num_images

    def get_size_image(self) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        return self._size_image

    def get_num_images(self) -> int:
        return self._num_images

    # def update_image_data(self, **kwargs):
    def update_image_data(self, in_shape_image: Tuple[int, ...]) -> None:
        raise NotImplementedError

    def _initialize_gendata(self) -> None:
        raise NotImplementedError

    def _update_gendata(self, **kwargs) -> None:
        raise NotImplementedError

    def _get_image(self, in_image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_image(self, in_image: np.ndarray, **kwargs) -> np.ndarray:
        self._update_gendata(**kwargs)
        out_image = self._get_image(in_image)
        return out_image

    def get_2images(self, in_image_1: np.ndarray, in_image_2: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        self._update_gendata(**kwargs)
        out_image_1 = self._get_image(in_image_1)
        out_image_2 = self._get_image(in_image_2)
        return (out_image_1, out_image_2)

    def get_many_images(self, in_list_images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
        self._update_gendata(**kwargs)
        out_list_images = []
        for in_image in in_list_images:
            out_image = self._get_image(in_image)
            out_list_images.append(out_image)
        return out_list_images

    def get_shape_output_image(self, in_shape_image: Tuple[int, ...]) -> Tuple[int, ...]:
        if ImagesUtil.is_without_channels(self._size_image, in_shape_image):
            return (self._num_images,) + self._size_image
        else:
            num_channels = ImagesUtil.get_num_channels(self._size_image, in_shape_image)
            return (self._num_images,) + self._size_image + (num_channels,)

    def compute_images_all(self, in_list_images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
        seed_0 = kwargs['seed_0']

        out_list_images = []
        for in_image in in_list_images:
            out_shape = self.get_shape_output_image(in_image.shape)
            out_image = np.ndarray(out_shape, dtype=in_image.dtype)
            out_list_images.append(out_image)

        for index in range(self._num_images):
            seed = self.update_seed_with_index(seed_0, index)
            add_kwargs = {'index': index, 'seed': seed}
            self._update_gendata(**add_kwargs)

            for i, in_image in in_list_images:
                out_list_images[i][index] = self._get_image(in_image)

        return out_list_images

    def get_text_description(self) -> str:
        raise NotImplementedError

    def update_seed_with_index(self, seed: int, index: int) -> Union[int, None]:
        if seed:
            return seed + index
        else:
            return None


class NullGenerator(ImageGenerator):

    def __init__(self) -> None:
        super(NullGenerator, self).__init__(size_image=(0, 0, 0),
                                            num_images=1)

    def update_image_data(self, in_shape_image: Tuple[int, ...]) -> None:
        pass

    def _initialize_gendata(self) -> None:
        pass

    def _update_gendata(self, **kwargs) -> None:
        pass

    def get_text_description(self) -> str:
        return ''

    def _get_image(self, in_image: np.ndarray) -> np.ndarray:
        return in_image


class CombinedImagesGenerator(ImageGenerator):

    def __init__(self, list_image_generators: List[ImageGenerator]) -> None:
        self._list_image_generators = list_image_generators

        size_image = list_image_generators[-1].get_size_image()
        num_images = self._get_compute_num_images()

        super(CombinedImagesGenerator, self).__init__(size_image, num_images)

    def update_image_data(self, in_shape_image: Tuple[int, ...]) -> None:
        for image_generator in self._list_image_generators:
            image_generator.update_image_data(in_shape_image)

        self._num_images = self._get_compute_num_images()

    def _initialize_gendata(self) -> None:
        for image_generator in self._list_image_generators:
            image_generator._initialize_gendata()

    def _update_gendata(self, **kwargs) -> None:
        for image_generator in self._list_image_generators:
            image_generator._update_gendata(**kwargs)

    def _get_image(self, in_image: np.ndarray) -> np.ndarray:
        out_image = in_image
        for image_generator in self._list_image_generators:
            out_image = image_generator._get_image(out_image)

        return out_image

    def _get_compute_num_images(self) -> int:
        num_images_prodrun = 1
        for image_generator in self._list_image_generators:
            num_images_prodrun *= image_generator.get_num_images()

        return num_images_prodrun

    def get_text_description(self) -> str:
        message = ''
        for image_generator in self._list_image_generators:
            message += image_generator.get_text_description()

        return message
