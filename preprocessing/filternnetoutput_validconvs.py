
from typing import Tuple
import numpy as np

#from common.constant import *
from common.exception_manager import catch_error_exception
from common.function_util import ImagesUtil
from imageoperators.boundingboxes import BoundingBoxes
from preprocessing.imagegenerator import ImageGenerator

BoundBoxNDType = Tuple[Tuple[int, int], ...]


class FilterNnetOutputValidConvs(ImageGenerator):

    _type_progression_outside_output_nnet = 'linear'
    _avail_type_progression_outside_output_nnet = ['linear', 'quadratic', 'cubic', 'exponential', 'all_outputs_Unet']

    def __init__(self,
                 size_image: Tuple[int, ...],
                 size_output_image: Tuple[int, ...],
                 is_multiple_outputs_nnet: bool = False
                 ) -> None:
        self._size_image = size_image
        self._size_output_image = size_output_image
        self._is_multiple_outputs_nnet = is_multiple_outputs_nnet

        if is_multiple_outputs_nnet:
            self._type_progression = 'all_outputs_Unet'
            if type(size_output_image) != type([]):
                message = 'Wrong input argument \'size_output_image\': \'%s\'. It must be a list of tuples'
                catch_error_exception(message)
        else:
            self._type_progression = 'quadratic'

        self._compute_probabilitymap_output_nnet()

        super(FilterNnetOutputValidConvs, self).__init__(size_image, 1)

    @staticmethod
    def _calc_tensor_product_2D(a: int, b: int) -> np.ndarray:
        return np.einsum('i,j->ij', a, b)

    @staticmethod
    def _calc_tensor_product_3D(a:int, b:int, c:int) -> np.ndarray:
        return np.einsum('i,j,k->ijk', a, b, c)

    @staticmethod
    def _calc_linear_progression(coord_0: int, coord_1:int) -> np.ndarray:
        return np.linspace(0, 1, coord_1 - coord_0)

    @staticmethod
    def _calc_quadratic_progression(coord_0: int, coord_1:int) -> np.ndarray:
        return np.power(np.linspace(0, 1, coord_1 - coord_0), 2)

    @staticmethod
    def _calc_cubic_progression(coord_0: int, coord_1: int) -> np.ndarray:
        return np.power(np.linspace(0, 1, coord_1 - coord_0), 3)

    @staticmethod
    def calc_exponential_progression(coord_0: int, coord_1: int) -> np.ndarray:
        return (np.exp(np.linspace(0, 1, coord_1 - coord_0)) - 1)/(np.exp(1) - 1)

    def update_image_data(self, in_shape_image: Tuple[int, ...]) -> None:
        self._num_images = in_shape_image[0]

    def get_probmap_output_nnet(self) -> np.ndarray:
        return self._probmap_output_nnet

    def _get_image(self, in_image: np.ndarray) -> np.ndarray:
        return self._get_filtered_probout_nnet(in_image)

    def _get_filtered_probout_nnet(self, in_image: np.ndarray) -> np.ndarray:
        if self._check_correct_dims_image_filter(in_image.shape):
            if ImagesUtil.is_image_without_channels(self._size_image, in_image.shape):
                return np.multiply(self._probmap_output_nnet, in_image)
            else:
                return np.multiply(self._probmap_output_nnet, in_image[..., :])
        else:
            return None

    def _compute_progression_increasing(self, coord_0: int, coord_1: int) -> np.ndarray:
        if self._type_progression == 'linear':
            return self._calc_linear_progression(coord_0, coord_1)
        elif self._type_progression == 'quadratic':
            return self._calc_quadratic_progression(coord_0, coord_1)
        elif self._type_progression == 'cubic':
            return self._calc_cubic_progression(coord_0, coord_1)
        elif self._type_progression == 'exponential':
            return self.calc_exponential_progression(coord_0, coord_1)
        elif self._type_progression == 'all_outputs_Unet':
            # assume piecewise quadratic progression
            return self._calc_quadratic_progression(coord_0, coord_1)
        else:
            return 0

    def _compute_progression_decreasing(self, coord_0: int, coord_1: int) -> np.ndarray:
        return self._compute_progression_increasing(coord_0, coord_1)[::-1]

    def _fill_flat_interior_bounding_box(self, inner_bounding_box: BoundBoxNDType) -> None:
        return NotImplemented

    def _fill_flat_exterior_bounding_box(self, outer_bounding_box: BoundBoxNDType) -> None:
        return NotImplemented

    def _fill_progression_between_two_bounding_boxes(self,
                                                     inner_bounding_box: BoundBoxNDType,
                                                     outer_bounding_box: BoundBoxNDType,
                                                     propa_value_in: float = 1.0,
                                                     propa_value_out: float = 0.0
                                                     ) -> None:
        return NotImplemented

    def _check_correct_dims_image_filter(self, in_shape_image: Tuple[int, ...]) -> bool:
        if ImagesUtil.is_image_without_channels(self._size_image, in_shape_image):
            in_size_image = in_shape_image
        else:
            in_size_image = in_shape_image[:-2]
        if in_size_image == self._size_image:
            return True
        else:
            return False

    def _compute_probabilitymap_output_nnet(self) -> None:
        self._probmap_output_nnet = np.zeros(self._size_image, dtype=np.float)

        if self._is_multiple_outputs_nnet:
            # set flat probability equal to 'one' inside inner output of nnet
            # set piecewise probability distribution in between bounding boxes corresponding to outputs of Unet up to 'max_depth'
            # in between bounding boxes, assume quadratic distribution in between two values with diff: 1/num_output_nnet
            num_outputs_nnet = len(self._size_output_image)
            sizes_bounding_boxes_output = self._size_output_image + [self._size_image]

            inner_bounding_box_output = BoundingBoxes.compute_bounding_box_centered_image_fit_image(self._size_image,
                                                                                                    sizes_bounding_boxes_output[0])
            self._fill_flat_interior_bounding_box(inner_bounding_box_output)

            for i in range(num_outputs_nnet):
                propa_value_in = 1.0 - i / float(num_outputs_nnet)
                propa_value_out = 1.0 - (i + 1) / float(num_outputs_nnet)

                inner_bounding_box_output = BoundingBoxes.compute_bounding_box_centered_image_fit_image(self._size_image,
                                                                                                        sizes_bounding_boxes_output[i])
                outer_bounding_box_output = BoundingBoxes.compute_bounding_box_centered_image_fit_image(self._size_image,
                                                                                                        sizes_bounding_boxes_output[i + 1])
                self._fill_progression_between_two_bounding_boxes(inner_bounding_box_output,
                                                                  outer_bounding_box_output,
                                                                  propa_value_in,
                                                                  propa_value_out)
        else:
            # set flat probability equal to 'one' inside output of nnet
            # set probability distribution (linear, quadratic, ...) in between output of nnet and borders of image
            inner_bounding_box_output = BoundingBoxes.compute_bounding_box_centered_image_fit_image(self._size_image,
                                                                                                    self._size_output_image)
            image_bounding_box_default = BoundingBoxes.get_default_bounding_box_image(self._size_image)

            self._fill_flat_interior_bounding_box(inner_bounding_box_output)
            self._fill_progression_between_two_bounding_boxes(inner_bounding_box_output,
                                                              image_bounding_box_default)


class FilteringNnetOutputValidConvs2D(FilterNnetOutputValidConvs):

    def __init__(self,
                 size_image: Tuple[int, int],
                 size_output_image: Tuple[int, int]
                 ) -> None:
        super(FilteringNnetOutputValidConvs2D, self).__init__(size_image, size_output_image)

    def _fill_flat_interior_bounding_box(self, inner_bounding_box: BoundBoxNDType) -> None:
        # assign probability 'one' inside box
        (x_left, x_right, y_down, y_up) = inner_bounding_box
        self._probmap_output_nnet[x_left:x_right, y_down:y_up] = 1.0

    def _fill_flat_exterior_bounding_box(self, outer_bounding_box: BoundBoxNDType) -> None:
        # assign probability 'zero' outside box
        (x_left, x_right, y_down, y_up) = outer_bounding_box
        self._probmap_output_nnet[0:x_left, :] = 0.0
        self._probmap_output_nnet[x_right:, :] = 0.0
        self._probmap_output_nnet[:, 0:y_down] = 0.0
        self._probmap_output_nnet[:, y_up:] = 0.0

    def _fill_progression_between_two_bounding_boxes(self,
                                                     inner_bounding_box: BoundBoxNDType,
                                                     outer_bounding_box: BoundBoxNDType,
                                                     propa_value_in: float = 1.0,
                                                     propa_value_out: float = 0.0
                                                     ) -> None:
        # assign probability distribution between 'inner' and 'outer' boundboxes, between values 'propa_value_in' and 'propa_value_out'
        (x_left_in, x_right_in, y_down_in, y_up_in) = inner_bounding_box
        (x_left_out, x_right_out, y_down_out, y_up_out) = outer_bounding_box

        progression_x_left = self._compute_progression_increasing(x_left_out, x_left_in) * (propa_value_in - propa_value_out) + propa_value_out
        progression_x_right = self._compute_progression_decreasing(x_right_in, x_right_out) * (propa_value_in - propa_value_out) + propa_value_out
        progression_y_down = self._compute_progression_increasing(y_down_out, y_down_in) * (propa_value_in - propa_value_out) + propa_value_out
        progression_y_up = self._compute_progression_decreasing(y_up_in, y_up_out) * (propa_value_in - propa_value_out) + propa_value_out
        progression_x_middle = np.ones([x_right_in - x_left_in]) * propa_value_in
        progression_y_middle = np.ones([y_up_in - y_down_in]) * propa_value_in

        # laterals
        self._probmap_output_nnet[x_left_out:x_left_in, y_down_in:y_up_in] = self._calc_tensor_product_2D(progression_x_left, progression_y_middle)
        self._probmap_output_nnet[x_right_in:x_right_out, y_down_in:y_up_in] = self._calc_tensor_product_2D(progression_x_right, progression_y_middle)
        self._probmap_output_nnet[x_left_in:x_right_in, y_down_out:y_down_in] = self._calc_tensor_product_2D(progression_x_middle, progression_y_down)
        self._probmap_output_nnet[x_left_in:x_right_in, y_up_in:y_up_out] = self._calc_tensor_product_2D(progression_x_middle, progression_y_up)
        # corners
        self._probmap_output_nnet[x_left_out:x_left_in, y_down_out:y_down_in] = self._calc_tensor_product_2D(progression_x_left, progression_y_down)
        self._probmap_output_nnet[x_right_in:x_right_out, y_down_out:y_down_in] = self._calc_tensor_product_2D(progression_x_right, progression_y_down)
        self._probmap_output_nnet[x_left_out:x_left_in, y_up_in:y_up_out] = self._calc_tensor_product_2D(progression_x_left, progression_y_up)
        self._probmap_output_nnet[x_right_in:x_right_out, y_up_in:y_up_out] = self._calc_tensor_product_2D(progression_x_right, progression_y_up)


class FilteringNnetOutputValidConvs3D(FilterNnetOutputValidConvs):

    def __init__(self,
                 size_image: Tuple[int, int, int],
                 size_output_image: Tuple[int, int, int]
                 ) -> None:
        super(FilteringNnetOutputValidConvs3D, self).__init__(size_image, size_output_image)

    def _fill_flat_interior_bounding_box(self, inner_bounding_box: BoundBoxNDType) -> None:
        # assign probability 'one' inside box
        (z_back, z_front, x_left, x_right, y_down, y_up) = inner_bounding_box
        self._probmap_output_nnet[z_back:z_front, x_left:x_right, y_down:y_up] = 1.0

    def _fill_flat_exterior_bounding_box(self, outer_bounding_box: BoundBoxNDType) -> None:
        # assign probability 'zero' outside box
        (z_back, z_front, x_left, x_right, y_down, y_up) = outer_bounding_box
        self._probmap_output_nnet[0:z_back, :, :] = 0.0
        self._probmap_output_nnet[z_front:, :, :] = 0.0
        self._probmap_output_nnet[:, 0:x_left, :] = 0.0
        self._probmap_output_nnet[:, x_right:, :] = 0.0
        self._probmap_output_nnet[:, :, 0:y_down] = 0.0
        self._probmap_output_nnet[:, :, y_up:] = 0.0

    def _fill_progression_between_two_bounding_boxes(self,
                                                     inner_bounding_box: BoundBoxNDType,
                                                     outer_bounding_box: BoundBoxNDType,
                                                     propa_value_in: float = 1.0,
                                                     propa_value_out: float = 0.0
                                                     ) -> None:
        # assign probability distribution between 'inner' and 'outer' boundboxes, between values 'propa_value_in' and 'propa_value_out'
        (z_back_in, z_front_in, x_left_in, x_right_in, y_down_in, y_up_in) = inner_bounding_box
        (z_back_out, z_front_out, x_left_out, x_right_out, y_down_out, y_up_out) = outer_bounding_box

        progression_z_back = self._compute_progression_increasing(z_back_out, z_back_in) * (propa_value_in - propa_value_out) + propa_value_out
        progression_z_front = self._compute_progression_decreasing(z_front_in, z_front_out) * (propa_value_in - propa_value_out) + propa_value_out
        progression_x_left = self._compute_progression_increasing(x_left_out, x_left_in) * (propa_value_in - propa_value_out) + propa_value_out
        progression_x_right = self._compute_progression_decreasing(x_right_in, x_right_out) * (propa_value_in - propa_value_out) + propa_value_out
        progression_y_down = self._compute_progression_increasing(y_down_out, y_down_in) * (propa_value_in - propa_value_out) + propa_value_out
        progression_y_up = self._compute_progression_decreasing(y_up_in, y_up_out) * (propa_value_in - propa_value_out) + propa_value_out
        progression_z_middle = np.ones([z_front_in - z_back_in]) * propa_value_in
        progression_x_middle = np.ones([x_right_in - x_left_in]) * propa_value_in
        progression_y_middle = np.ones([y_up_in - y_down_in]) * propa_value_in

        # laterals
        self._probmap_output_nnet[z_back_in:z_front_in, x_left_out:x_left_in, y_down_in:y_up_in] = self._calc_tensor_product_3D(progression_z_middle, progression_x_left, progression_y_middle)
        self._probmap_output_nnet[z_back_in:z_front_in, x_right_in:x_right_out, y_down_in:y_up_in] = self._calc_tensor_product_3D(progression_z_middle, progression_x_right, progression_y_middle)
        self._probmap_output_nnet[z_back_in:z_front_in, x_left_in:x_right_in, y_down_out:y_down_in] = self._calc_tensor_product_3D(progression_z_middle, progression_x_middle, progression_y_down)
        self._probmap_output_nnet[z_back_in:z_front_in, x_left_in:x_right_in, y_up_in:y_up_out] = self._calc_tensor_product_3D(progression_z_middle, progression_x_middle, progression_y_up)
        self._probmap_output_nnet[z_back_out:z_back_in, x_left_in:x_right_in, y_down_in:y_up_in] = self._calc_tensor_product_3D(progression_z_back, progression_x_middle, progression_y_middle)
        self._probmap_output_nnet[z_front_in:z_front_out, x_left_in:x_right_in, y_down_in:y_up_in] = self._calc_tensor_product_3D(progression_z_front, progression_x_middle, progression_y_middle)
        # edges corners
        self._probmap_output_nnet[z_back_out:z_back_in, x_left_out:x_left_in, y_down_in:y_up_in] = self._calc_tensor_product_3D(progression_z_back, progression_x_left, progression_y_middle)
        self._probmap_output_nnet[z_back_out:z_back_in, x_right_in:x_right_out, y_down_in:y_up_in] = self._calc_tensor_product_3D(progression_z_back, progression_x_right, progression_y_middle)
        self._probmap_output_nnet[z_back_out:z_back_in, x_left_in:x_right_in, y_down_out:y_down_in] = self._calc_tensor_product_3D(progression_z_back, progression_x_middle, progression_y_down)
        self._probmap_output_nnet[z_back_out:z_back_in, x_left_in:x_right_in, y_up_in:y_up_out] = self._calc_tensor_product_3D(progression_z_back, progression_x_middle, progression_y_up)
        self._probmap_output_nnet[z_front_in:z_front_out, x_left_out:x_left_in, y_down_in:y_up_in] = self._calc_tensor_product_3D(progression_z_front, progression_x_left, progression_y_middle)
        self._probmap_output_nnet[z_front_in:z_front_out, x_right_in:x_right_out, y_down_in:y_up_in] = self._calc_tensor_product_3D(progression_z_front, progression_x_right, progression_y_middle)
        self._probmap_output_nnet[z_front_in:z_front_out, x_left_in:x_right_in, y_down_out:y_down_in] = self._calc_tensor_product_3D(progression_z_front, progression_x_middle, progression_y_down)
        self._probmap_output_nnet[z_front_in:z_front_out, x_left_in:x_right_in, y_up_in:y_up_out] = self._calc_tensor_product_3D(progression_z_front, progression_x_middle, progression_y_up)
        self._probmap_output_nnet[z_back_in:z_front_in, x_left_out:x_left_in, y_down_out:y_down_in] = self._calc_tensor_product_3D(progression_z_middle, progression_x_left, progression_y_down)
        self._probmap_output_nnet[z_back_in:z_front_in, x_right_in:x_right_out, y_down_out:y_down_in] = self._calc_tensor_product_3D(progression_z_middle, progression_x_right, progression_y_down)
        self._probmap_output_nnet[z_back_in:z_front_in, x_left_out:x_left_in, y_up_in:y_up_out] = self._calc_tensor_product_3D(progression_z_middle, progression_x_left, progression_y_up)
        self._probmap_output_nnet[z_back_in:z_front_in, x_right_in:x_right_out, y_up_in:y_up_out] = self._calc_tensor_product_3D(progression_z_middle, progression_x_right, progression_y_up)
        # corners
        self._probmap_output_nnet[z_back_out:z_back_in, x_left_out:x_left_in, y_down_out:y_down_in] = self._calc_tensor_product_3D(progression_z_back, progression_x_left, progression_y_down)
        self._probmap_output_nnet[z_back_out:z_back_in, x_right_in:x_right_out, y_down_out:y_down_in] = self._calc_tensor_product_3D(progression_z_back, progression_x_right, progression_y_down)
        self._probmap_output_nnet[z_back_out:z_back_in, x_left_out:x_left_in, y_up_in:y_up_out] = self._calc_tensor_product_3D(progression_z_back, progression_x_left, progression_y_up)
        self._probmap_output_nnet[z_back_out:z_back_in, x_right_in:x_right_out, y_up_in:y_up_out] = self._calc_tensor_product_3D(progression_z_back, progression_x_right, progression_y_up)
        self._probmap_output_nnet[z_front_in:z_front_out, x_left_out:x_left_in, y_down_out:y_down_in] = self._calc_tensor_product_3D(progression_z_front, progression_x_left, progression_y_down)
        self._probmap_output_nnet[z_front_in:z_front_out, x_right_in:x_right_out, y_down_out:y_down_in] = self._calc_tensor_product_3D(progression_z_front, progression_x_right, progression_y_down)
        self._probmap_output_nnet[z_front_in:z_front_out, x_left_out:x_left_in, y_up_in:y_up_out] = self._calc_tensor_product_3D(progression_z_front, progression_x_left, progression_y_up)
        self._probmap_output_nnet[z_front_in:z_front_out, x_right_in:x_right_out, y_up_in:y_up_out] = self._calc_tensor_product_3D(progression_z_front, progression_x_right, progression_y_up)