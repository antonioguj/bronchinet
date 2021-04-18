
from typing import Tuple, List, Union
import numpy as np

from scipy.ndimage import map_coordinates, gaussian_filter
from elasticdeform import deform_random_grid

from common.exceptionmanager import catch_error_exception
from preprocessing.imagegenerator import ImageGenerator


class ElasticDeformImages(ImageGenerator):
    _order_interp_image = 3
    _order_interp_mask = 0

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 fill_mode: str = 'nearest',
                 cval: float = 0.0
                 ) -> None:
        super(ElasticDeformImages, self).__init__(size_image, num_images=1)

        self._fill_mode = fill_mode
        self._cval = cval
        self._ndims = len(size_image)

        if (self._ndims != 2) and (self._ndims != 3):
            message = 'ElasticDeformImages:__init__: wrong \'ndims\': %s' % (self._ndims)
            catch_error_exception(message)

        self._initialize_gendata()

    def update_image_data(self, in_shape_image: Tuple[int, ...]) -> None:
        # self._num_images = in_shape_image[0]
        pass

    def _initialize_gendata(self) -> None:
        self._gendata_elastic_deform = None
        self._count_trans_in_images = 0

    def _update_gendata(self, **kwargs) -> None:
        seed = kwargs['seed']
        self._gendata_elastic_deform = self._get_calcgendata_elastic_deform(seed)
        self._count_trans_in_images = 0

    def _get_image(self, in_image: np.ndarray) -> np.ndarray:
        is_type_input_image = (self._count_trans_in_images == 0)
        self._count_trans_in_images += 1
        return self._get_transformed_image(in_image, is_type_input_image)

    def _get_transformed_image(self, in_image: np.ndarray, is_type_input_image: bool = False) -> np.ndarray:
        if is_type_input_image:
            return map_coordinates(in_image, self._gendata_elastic_deform, order=self._order_interp_image,
                                   mode=self._fill_mode, cval=self._cval).reshape(self._size_image)
        else:
            return map_coordinates(in_image, self._gendata_elastic_deform, order=self._order_interp_mask,
                                   mode=self._fill_mode, cval=self._cval).reshape(self._size_image)

    def _get_inverse_transformed_image(self, in_image: np.ndarray, is_type_input_image: bool = False) -> np.ndarray:
        message = 'Inverse transformation not implemented for Elastic Deformations'
        catch_error_exception(message)

    def _get_calcgendata_elastic_deform(self, seed: int = None) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def _get_type_elastic_deform(cls) -> str:
        raise NotImplementedError

    def get_text_description(self) -> str:
        message = 'Elastic deformations of images...\n'
        message += '- type of elastic deformation: \'%s\'...\n' % (self._get_type_elastic_deform())
        return message


class ElasticDeformGridwiseImages(ElasticDeformImages):
    _sigma_default = 25
    _points_default = 3
    _type_elastic_deform = 'Grid-wise'

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 sigma: int = _sigma_default,
                 points: int = _points_default,
                 fill_mode: str = 'nearest',
                 cval: float = 0.0,
                 ) -> None:
        self._sigma = sigma
        self._points = points

        super(ElasticDeformGridwiseImages, self).__init__(size_image, fill_mode=fill_mode, cval=cval)

    def _get_calcgendata_elastic_deform(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        if self._ndims == 2:
            # creates the grid of coordinates of the points of the image (a ndim array per dimension)
            coordinates = np.meshgrid(np.arange(self._size_image[0]),
                                      np.arange(self._size_image[1]), indexing='ij')
            # creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
            xi = np.meshgrid(np.linspace(0, self._points - 1, self._size_image[0]),
                             np.linspace(0, self._points - 1, self._size_image[1]), indexing='ij')
            grid = [self._points, self._points]

        elif self._ndims == 3:
            # creates the grid of coordinates of the points of the image (a ndim array per dimension)
            coordinates = np.meshgrid(np.arange(self._size_image[0]),
                                      np.arange(self._size_image[1]),
                                      np.arange(self._size_image[2]), indexing='ij')
            # creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
            xi = np.meshgrid(np.linspace(0, self._points - 1, self._size_image[0]),
                             np.linspace(0, self._points - 1, self._size_image[1]),
                             np.linspace(0, self._points - 1, self._size_image[2]), indexing='ij')
            grid = [self._points, self._points, self._points]

        # creates the deformation along each dimension and then add it to the coordinates
        for i in range(self._ndims):
            # creating the displacement at the control points
            yi = np.random.randn(*grid) * self._sigma
            # print(y.shape,coordinates[i].shape) #y and coordinates[i] should be of the same shape,
            # otherwise the same displacement is applied to every ?row? of points ?
            y = map_coordinates(yi, xi, order=3).reshape(self._size_image)
            # adding the displacement
            coordinates[i] = np.add(coordinates[i], y)
        # endfor

        return np.asarray(coordinates)

    @classmethod
    def _get_type_elastic_deform(cls) -> str:
        return cls._type_elastic_deform


class ElasticDeformPixelwiseImages(ElasticDeformImages):
    _alpha_default = 15
    _sigma_default = 3
    _type_elastic_deform = 'Pixel-wise'

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 alpha: int = _alpha_default,
                 sigma: int = _sigma_default,
                 fill_mode: str = 'nearest',
                 cval: float = 0.0,
                 ) -> None:
        self._alpha = alpha
        self._sigma = sigma

        super(ElasticDeformPixelwiseImages, self).__init__(size_image, fill_mode=fill_mode, cval=cval)

    def _get_calcgendata_elastic_deform(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        if self._ndims == 2:
            xi_dirs = np.meshgrid(np.arange(self._size_image[0]),
                                  np.arange(self._size_image[1]), indexing='ij')
        elif self._ndims == 3:
            xi_dirs = np.meshgrid(np.arange(self._size_image[0]),
                                  np.arange(self._size_image[1]),
                                  np.arange(self._size_image[2]), indexing='ij')

        indices = []
        for i in range(self._ndims):
            # originally with random_state.rand * 2 - 1
            dx_i = gaussian_filter(np.random.randn(*self._size_image),
                                   self._sigma, mode='constant', cval=0) * self._alpha
            indices.append(xi_dirs[i] + dx_i)
        # endfor

        return np.asarray(indices)

    @classmethod
    def _get_type_elastic_deform(cls) -> str:
        return cls._type_elastic_deform


class ElasticDeformGridwiseImagesGijs(ElasticDeformImages):
    _sigma_default = 25
    _points_default = 3
    _type_elastic_deform = 'Grid-wise_Gijs'

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 sigma: int = _sigma_default,
                 points: int = _points_default,
                 fill_mode: str = 'nearest',
                 cval: float = 0.0,
                 ) -> None:
        self._sigma = sigma
        self._points = points

        super(ElasticDeformGridwiseImagesGijs, self).__init__(size_image, fill_mode=fill_mode, cval=cval)

    def _get_calcgendata_elastic_deform(self, seed: int = None) -> np.ndarray:
        pass

    def _get_image(self, in_image: np.ndarray) -> np.ndarray:
        out_image = deform_random_grid(in_image, sigma=self._sigma, points=self._points, order=self._order_interp_image,
                                       mode=self._fill_mode, cval=self._cval)
        return out_image

    def get_2images(self, in_image_1: np.ndarray, in_image_2: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        (out_image_1, out_image_2) = deform_random_grid([in_image_1, in_image_2],
                                                        sigma=self._sigma,
                                                        points=self._points,
                                                        order=[self._order_interp_image, self._order_interp_mask],
                                                        mode=self._fill_mode, cval=self._cval)
        return (out_image_1, out_image_2)

    def get_many_images(self, in_list_images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
        out_list_images = deform_random_grid(in_list_images,
                                             sigma=self._sigma,
                                             points=self._points,
                                             order=[self._order_interp_image]
                                             + [self._order_interp_mask] * (len(in_list_images) - 1),
                                             mode=self._fill_mode, cval=self._cval)
        return out_list_images

    @classmethod
    def _get_type_elastic_deform(cls) -> str:
        return cls._type_elastic_deform
