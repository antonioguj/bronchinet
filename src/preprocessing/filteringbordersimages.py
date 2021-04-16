
from typing import Tuple, List, Union
import numpy as np

from common.exceptionmanager import catch_error_exception
from common.functionutil import ImagesUtil
from imageoperators.boundingboxes import BoundingBoxes, BoundBox3DType, BoundBox2DType
from preprocessing.imagegenerator import ImageGenerator

TYPES_FILTERING_AVAIL = ['linear', 'quadratic', 'cubic', 'exponential']


class FilteringBordersImages(ImageGenerator):
    _type_filtering_default = 'quadratic'

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 size_output_image: Union[Tuple[int, int, int], Tuple[int, int],
                                          List[Tuple[int, int, int]], List[Tuple[int, int]]],
                 type_filtering: str = _type_filtering_default,
                 is_filtering_multiple_windows: bool = False
                 ) -> None:
        self._size_image = size_image
        self._ndims = len(size_image)
        self._size_output_image = size_output_image
        self._type_filtering = type_filtering
        self._is_filtering_multiple_windows = is_filtering_multiple_windows

        if self._ndims == 2:
            self._func_multiply_matrixes_channels = self._multiply_matrixes_with_channels_2d
        elif self._ndims == 3:
            self._func_multiply_matrixes_channels = self._multiply_matrixes_with_channels_3d
        else:
            message = 'FilteringBorderEffectsImages:__init__: wrong \'ndims\': %s...' % (self._ndims)
            catch_error_exception(message)

        self._compute_factor_filtering()

        super(FilteringBordersImages, self).__init__(size_image, 1)

    def update_image_data(self, in_shape_image: Tuple[int, ...]) -> None:
        self._num_images = in_shape_image[0]

    def _get_image(self, in_image: np.ndarray) -> np.ndarray:
        return self._get_filtered_image(in_image)

    def _get_filtered_image(self, in_image: np.ndarray) -> np.ndarray:
        if ImagesUtil.is_without_channels(self._size_image, in_image.shape):
            return np.multiply(in_image, self._factor_filtering)
        else:
            return self._func_multiply_matrixes_channels(in_image, self._factor_filtering)

    def _compute_progression_increasing(self, coord_0: int, coord_1: int) -> np.ndarray:
        if self._type_filtering == 'linear':
            return self._calc_linear_progression(coord_0, coord_1)
        elif self._type_filtering == 'quadratic':
            return self._calc_quadratic_progression(coord_0, coord_1)
        elif self._type_filtering == 'cubic':
            return self._calc_cubic_progression(coord_0, coord_1)
        elif self._type_filtering == 'exponential':
            return self.calc_exponential_progression(coord_0, coord_1)

    def _compute_progression_decreasing(self, coord_0: int, coord_1: int) -> np.ndarray:
        return self._compute_progression_increasing(coord_0, coord_1)[::-1]

    def _fill_flat_interior_boundbox(self, inner_boundbox: Union[BoundBox3DType, BoundBox2DType]) -> None:
        raise NotImplementedError

    def _fill_flat_exterior_boundbox(self, outer_boundbox: Union[BoundBox3DType, BoundBox2DType]) -> None:
        raise NotImplementedError

    def _fill_progression_between_two_boundboxes(self, inner_boundbox: Union[BoundBox3DType, BoundBox2DType],
                                                 outer_boundbox: Union[BoundBox3DType, BoundBox2DType],
                                                 propa_value_in: float = 1.0,
                                                 propa_value_out: float = 0.0) -> None:
        raise NotImplementedError

    @staticmethod
    def _multiply_matrixes_with_channels_2d(matrix_1_channels: np.ndarray, matrix_2: np.ndarray) -> np.ndarray:
        return np.einsum('ijk,ij->ijk', matrix_1_channels, matrix_2)

    @staticmethod
    def _multiply_matrixes_with_channels_3d(matrix_1_channels: np.ndarray, matrix_2: np.ndarray) -> np.ndarray:
        return np.einsum('ijkl,ijk->ijkl', matrix_1_channels, matrix_2)

    @staticmethod
    def _calc_tensor_product_2d(vector_1: np.ndarray, vector_2: np.ndarray) -> np.ndarray:
        return np.einsum('i,j->ij', vector_1, vector_2)

    @staticmethod
    def _calc_tensor_product_3d(vector_1: np.ndarray, vector_2: np.ndarray, vector_3: np.ndarray) -> np.ndarray:
        return np.einsum('i,j,k->ijk', vector_1, vector_2, vector_3)

    @staticmethod
    def _calc_linear_progression(coord_0: int, coord_1: int) -> np.ndarray:
        return np.linspace(0, 1, coord_1 - coord_0)

    @staticmethod
    def _calc_quadratic_progression(coord_0: int, coord_1: int) -> np.ndarray:
        return np.power(np.linspace(0, 1, coord_1 - coord_0), 2)

    @staticmethod
    def _calc_cubic_progression(coord_0: int, coord_1: int) -> np.ndarray:
        return np.power(np.linspace(0, 1, coord_1 - coord_0), 3)

    @staticmethod
    def calc_exponential_progression(coord_0: int, coord_1: int) -> np.ndarray:
        return (np.exp(np.linspace(0, 1, coord_1 - coord_0)) - 1) / (np.exp(1) - 1)

    def _compute_factor_filtering(self) -> None:
        self._factor_filtering = np.zeros(self._size_image, dtype=np.float32)

        if self._is_filtering_multiple_windows:
            # 'factor_filtering' defined:
            # - consider several concentrical windows, of increasing sizes, and until the input image border
            # - i_window == 0:
            #   - inside the window -> '1'
            #   - outside the window, and until next window -> decreasing function (linear, quadratic, ...)
            # - for i_window == 1, ...:
            #   - inside the window -> factor previous window
            #   - outside the window, and until next window (or input image borders) -> decreasing function
            num_windows = len(self._size_output_image)
            sizes_windows = self._size_output_image + [self._size_image]

            boundbox_output_image = BoundingBoxes.calc_boundbox_centered_image_fitimg(self._size_image,
                                                                                      sizes_windows[0])
            self._fill_flat_interior_boundbox(boundbox_output_image)

            for iwin in range(num_windows):
                iwindow_value_inner = 1.0 - iwin / float(num_windows)
                iwindow_value_outer = 1.0 - (iwin + 1) / float(num_windows)

                iwindow_boundbox_inner = BoundingBoxes.calc_boundbox_centered_image_fitimg(sizes_windows[iwin],
                                                                                           self._size_image)
                iwindow_boundbox_outer = BoundingBoxes.calc_boundbox_centered_image_fitimg(sizes_windows[iwin + 1],
                                                                                           self._size_image)
                self._fill_progression_between_two_boundboxes(iwindow_boundbox_inner, iwindow_boundbox_outer,
                                                              iwindow_value_inner, iwindow_value_outer)
        else:
            # 'factor_filtering' defined:
            # - inside the output window -> '1'
            # - outside the output window and until input image borders -> decreasing function (linear, quadratic, ...)
            boundbox_output_image = BoundingBoxes.calc_boundbox_centered_image_fitimg(self._size_output_image,
                                                                                      self._size_image)
            boundbox_input_image = BoundingBoxes.get_default_boundbox_image(self._size_image)

            self._fill_flat_interior_boundbox(boundbox_output_image)
            self._fill_progression_between_two_boundboxes(boundbox_output_image, boundbox_input_image)


class FilteringBordersImages2D(FilteringBordersImages):

    def __init__(self,
                 size_image: Tuple[int, int],
                 size_output_image: Tuple[int, int]
                 ) -> None:
        super(FilteringBordersImages2D, self).__init__(size_image, size_output_image)

    def _fill_flat_interior_boundbox(self, inner_boundbox: BoundBox2DType) -> None:
        # set 'one' inside bounding-box
        ((x_left, x_right), (y_down, y_up)) = inner_boundbox
        self._factor_filtering[x_left:x_right, y_down:y_up] = 1.0

    def _fill_flat_exterior_boundbox(self, outer_boundbox: BoundBox2DType) -> None:
        # set 'zero' outside bounding-box
        ((x_left, x_right), (y_down, y_up)) = outer_boundbox
        self._factor_filtering[0:x_left, :] = 0.0
        self._factor_filtering[x_right:, :] = 0.0
        self._factor_filtering[:, 0:y_down] = 0.0
        self._factor_filtering[:, y_up:] = 0.0

    def _fill_progression_between_two_boundboxes(self, inner_boundbox: BoundBox2DType, outer_boundbox: BoundBox2DType,
                                                 value_inner: float = 1.0, value_outer: float = 0.0
                                                 ) -> None:
        # set progression between 'value_inner' and 'value_outer', between 'inner' and 'outer' bounding-boxes
        ((x_left_in, x_right_in), (y_down_in, y_up_in)) = inner_boundbox
        ((x_left_out, x_right_out), (y_down_out, y_up_out)) = outer_boundbox

        progression_x_left = \
            self._compute_progression_increasing(x_left_out, x_left_in) * (value_inner - value_outer) + value_outer
        progression_x_right = \
            self._compute_progression_decreasing(x_right_in, x_right_out) * (value_inner - value_outer) + value_outer
        progression_y_down = \
            self._compute_progression_increasing(y_down_out, y_down_in) * (value_inner - value_outer) + value_outer
        progression_y_up = \
            self._compute_progression_decreasing(y_up_in, y_up_out) * (value_inner - value_outer) + value_outer
        progression_x_middle = np.ones([x_right_in - x_left_in]) * value_inner
        progression_y_middle = np.ones([y_up_in - y_down_in]) * value_inner

        # laterals
        self._factor_filtering[x_left_out:x_left_in, y_down_in:y_up_in] = \
            self._calc_tensor_product_2d(progression_x_left, progression_y_middle)
        self._factor_filtering[x_right_in:x_right_out, y_down_in:y_up_in] = \
            self._calc_tensor_product_2d(progression_x_right, progression_y_middle)
        self._factor_filtering[x_left_in:x_right_in, y_down_out:y_down_in] = \
            self._calc_tensor_product_2d(progression_x_middle, progression_y_down)
        self._factor_filtering[x_left_in:x_right_in, y_up_in:y_up_out] = \
            self._calc_tensor_product_2d(progression_x_middle, progression_y_up)
        # corners
        self._factor_filtering[x_left_out:x_left_in, y_down_out:y_down_in] = \
            self._calc_tensor_product_2d(progression_x_left, progression_y_down)
        self._factor_filtering[x_right_in:x_right_out, y_down_out:y_down_in] = \
            self._calc_tensor_product_2d(progression_x_right, progression_y_down)
        self._factor_filtering[x_left_out:x_left_in, y_up_in:y_up_out] = \
            self._calc_tensor_product_2d(progression_x_left, progression_y_up)
        self._factor_filtering[x_right_in:x_right_out, y_up_in:y_up_out] = \
            self._calc_tensor_product_2d(progression_x_right, progression_y_up)


class FilteringBordersImages3D(FilteringBordersImages):

    def __init__(self,
                 size_image: Tuple[int, int, int],
                 size_output_image: Tuple[int, int, int]
                 ) -> None:
        super(FilteringBordersImages3D, self).__init__(size_image, size_output_image)

    def _fill_flat_interior_boundbox(self, inner_boundbox: BoundBox3DType) -> None:
        # set 'one' inside bounding-box
        ((z_back, z_front), (x_left, x_right), (y_down, y_up)) = inner_boundbox
        self._factor_filtering[z_back:z_front, x_left:x_right, y_down:y_up] = 1.0

    def _fill_flat_exterior_boundbox(self, outer_boundbox: BoundBox3DType) -> None:
        # set 'zero' outside bounding-box
        ((z_back, z_front), (x_left, x_right), (y_down, y_up)) = outer_boundbox
        self._factor_filtering[0:z_back, :, :] = 0.0
        self._factor_filtering[z_front:, :, :] = 0.0
        self._factor_filtering[:, 0:x_left, :] = 0.0
        self._factor_filtering[:, x_right:, :] = 0.0
        self._factor_filtering[:, :, 0:y_down] = 0.0
        self._factor_filtering[:, :, y_up:] = 0.0

    def _fill_progression_between_two_boundboxes(self, inner_boundbox: BoundBox3DType, outer_boundbox: BoundBox3DType,
                                                 value_inner: float = 1.0, value_outer: float = 0.0
                                                 ) -> None:
        # set progression between 'value_inner' and 'value_outer', between 'inner' and 'outer' bounding-boxes
        ((z_back_in, z_front_in), (x_left_in, x_right_in), (y_down_in, y_up_in)) = inner_boundbox
        ((z_back_out, z_front_out), (x_left_out, x_right_out), (y_down_out, y_up_out)) = outer_boundbox

        progression_z_back = \
            self._compute_progression_increasing(z_back_out, z_back_in) * (value_inner - value_outer) + value_outer
        progression_z_front = \
            self._compute_progression_decreasing(z_front_in, z_front_out) * (value_inner - value_outer) + value_outer
        progression_x_left = \
            self._compute_progression_increasing(x_left_out, x_left_in) * (value_inner - value_outer) + value_outer
        progression_x_right = \
            self._compute_progression_decreasing(x_right_in, x_right_out) * (value_inner - value_outer) + value_outer
        progression_y_down = \
            self._compute_progression_increasing(y_down_out, y_down_in) * (value_inner - value_outer) + value_outer
        progression_y_up = \
            self._compute_progression_decreasing(y_up_in, y_up_out) * (value_inner - value_outer) + value_outer
        progression_z_middle = np.ones([z_front_in - z_back_in]) * value_inner
        progression_x_middle = np.ones([x_right_in - x_left_in]) * value_inner
        progression_y_middle = np.ones([y_up_in - y_down_in]) * value_inner

        # laterals
        self._factor_filtering[z_back_in:z_front_in, x_left_out:x_left_in, y_down_in:y_up_in] = \
            self._calc_tensor_product_3d(progression_z_middle, progression_x_left, progression_y_middle)
        self._factor_filtering[z_back_in:z_front_in, x_right_in:x_right_out, y_down_in:y_up_in] = \
            self._calc_tensor_product_3d(progression_z_middle, progression_x_right, progression_y_middle)
        self._factor_filtering[z_back_in:z_front_in, x_left_in:x_right_in, y_down_out:y_down_in] = \
            self._calc_tensor_product_3d(progression_z_middle, progression_x_middle, progression_y_down)
        self._factor_filtering[z_back_in:z_front_in, x_left_in:x_right_in, y_up_in:y_up_out] = \
            self._calc_tensor_product_3d(progression_z_middle, progression_x_middle, progression_y_up)
        self._factor_filtering[z_back_out:z_back_in, x_left_in:x_right_in, y_down_in:y_up_in] = \
            self._calc_tensor_product_3d(progression_z_back, progression_x_middle, progression_y_middle)
        self._factor_filtering[z_front_in:z_front_out, x_left_in:x_right_in, y_down_in:y_up_in] = \
            self._calc_tensor_product_3d(progression_z_front, progression_x_middle, progression_y_middle)
        # edges corners
        self._factor_filtering[z_back_out:z_back_in, x_left_out:x_left_in, y_down_in:y_up_in] = \
            self._calc_tensor_product_3d(progression_z_back, progression_x_left, progression_y_middle)
        self._factor_filtering[z_back_out:z_back_in, x_right_in:x_right_out, y_down_in:y_up_in] = \
            self._calc_tensor_product_3d(progression_z_back, progression_x_right, progression_y_middle)
        self._factor_filtering[z_back_out:z_back_in, x_left_in:x_right_in, y_down_out:y_down_in] = \
            self._calc_tensor_product_3d(progression_z_back, progression_x_middle, progression_y_down)
        self._factor_filtering[z_back_out:z_back_in, x_left_in:x_right_in, y_up_in:y_up_out] = \
            self._calc_tensor_product_3d(progression_z_back, progression_x_middle, progression_y_up)
        self._factor_filtering[z_front_in:z_front_out, x_left_out:x_left_in, y_down_in:y_up_in] = \
            self._calc_tensor_product_3d(progression_z_front, progression_x_left, progression_y_middle)
        self._factor_filtering[z_front_in:z_front_out, x_right_in:x_right_out, y_down_in:y_up_in] = \
            self._calc_tensor_product_3d(progression_z_front, progression_x_right, progression_y_middle)
        self._factor_filtering[z_front_in:z_front_out, x_left_in:x_right_in, y_down_out:y_down_in] = \
            self._calc_tensor_product_3d(progression_z_front, progression_x_middle, progression_y_down)
        self._factor_filtering[z_front_in:z_front_out, x_left_in:x_right_in, y_up_in:y_up_out] = \
            self._calc_tensor_product_3d(progression_z_front, progression_x_middle, progression_y_up)
        self._factor_filtering[z_back_in:z_front_in, x_left_out:x_left_in, y_down_out:y_down_in] = \
            self._calc_tensor_product_3d(progression_z_middle, progression_x_left, progression_y_down)
        self._factor_filtering[z_back_in:z_front_in, x_right_in:x_right_out, y_down_out:y_down_in] = \
            self._calc_tensor_product_3d(progression_z_middle, progression_x_right, progression_y_down)
        self._factor_filtering[z_back_in:z_front_in, x_left_out:x_left_in, y_up_in:y_up_out] = \
            self._calc_tensor_product_3d(progression_z_middle, progression_x_left, progression_y_up)
        self._factor_filtering[z_back_in:z_front_in, x_right_in:x_right_out, y_up_in:y_up_out] = \
            self._calc_tensor_product_3d(progression_z_middle, progression_x_right, progression_y_up)
        # corners
        self._factor_filtering[z_back_out:z_back_in, x_left_out:x_left_in, y_down_out:y_down_in] = \
            self._calc_tensor_product_3d(progression_z_back, progression_x_left, progression_y_down)
        self._factor_filtering[z_back_out:z_back_in, x_right_in:x_right_out, y_down_out:y_down_in] = \
            self._calc_tensor_product_3d(progression_z_back, progression_x_right, progression_y_down)
        self._factor_filtering[z_back_out:z_back_in, x_left_out:x_left_in, y_up_in:y_up_out] = \
            self._calc_tensor_product_3d(progression_z_back, progression_x_left, progression_y_up)
        self._factor_filtering[z_back_out:z_back_in, x_right_in:x_right_out, y_up_in:y_up_out] = \
            self._calc_tensor_product_3d(progression_z_back, progression_x_right, progression_y_up)
        self._factor_filtering[z_front_in:z_front_out, x_left_out:x_left_in, y_down_out:y_down_in] = \
            self._calc_tensor_product_3d(progression_z_front, progression_x_left, progression_y_down)
        self._factor_filtering[z_front_in:z_front_out, x_right_in:x_right_out, y_down_out:y_down_in] = \
            self._calc_tensor_product_3d(progression_z_front, progression_x_right, progression_y_down)
        self._factor_filtering[z_front_in:z_front_out, x_left_out:x_left_in, y_up_in:y_up_out] = \
            self._calc_tensor_product_3d(progression_z_front, progression_x_left, progression_y_up)
        self._factor_filtering[z_front_in:z_front_out, x_right_in:x_right_out, y_up_in:y_up_out] = \
            self._calc_tensor_product_3d(progression_z_front, progression_x_right, progression_y_up)
