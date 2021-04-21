
from typing import Tuple, Dict, Any, Union, Callable
import numpy as np
import scipy.ndimage as ndi

from common.exceptionmanager import catch_error_exception
from common.functionutil import ImagesUtil
from preprocessing.imagegenerator import ImageGenerator

_epsilon = 1e-6


class TransformRigidImages(ImageGenerator):

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 is_normalize_data: bool = False,
                 type_normalize_data: str = 'samplewise',
                 is_zca_whitening: bool = False,
                 is_inverse_transform: bool = False,
                 rescale_factor: float = None,
                 preprocessing_function: Callable[[np.ndarray], np.ndarray] = None
                 ) -> None:
        super(TransformRigidImages, self).__init__(size_image, num_images=1)

        if is_normalize_data:
            if type_normalize_data == 'featurewise':
                self._featurewise_center = True
                self._featurewise_std_normalization = True
                self._samplewise_center = False
                self._samplewise_std_normalization = False
            else:
                # type_normalize_data == 'samplewise'
                self._featurewise_center = False
                self._featurewise_std_normalization = False
                self._samplewise_center = True
                self._samplewise_std_normalization = True
        else:
            self._featurewise_center = False
            self._featurewise_std_normalization = False
            self._samplewise_center = False
            self._samplewise_std_normalization = False

        self._is_zca_whitening = is_zca_whitening
        self._zca_epsilon = 1e-6
        self._rescale_factor = rescale_factor
        self._preprocessing_function = preprocessing_function

        self._mean = None
        self._std = None
        self._principal_components = None

        self._is_inverse_transform = is_inverse_transform
        self._initialize_gendata()

    def update_image_data(self, in_shape_image: Tuple[int, ...]) -> None:
        # self._num_images = in_shape_image[0]
        pass

    def _initialize_gendata(self) -> None:
        self._transform_matrix = None
        self._transform_params = None
        self._count_trans_in_images = 0

    def _update_gendata(self, **kwargs) -> None:
        seed = kwargs['seed']
        (self._transform_matrix, self._transform_params) = self._calc_gendata_random_transform(seed)
        self._count_trans_in_images = 0

    def _get_image(self, in_image: np.ndarray) -> np.ndarray:
        is_type_input_image = (self._count_trans_in_images == 0)
        self._count_trans_in_images += 1
        return self._get_transformed_image(in_image, is_type_input_image=is_type_input_image)

    def _get_transformed_image(self, in_image: np.ndarray, is_type_input_image: bool = False) -> np.ndarray:
        if ImagesUtil.is_without_channels(self._size_image, in_image.shape):
            in_image = np.expand_dims(in_image, axis=-1)
            is_reshape_input_image = True
        else:
            is_reshape_input_image = False

        in_image = self._calc_transformed_image(in_image, is_type_input_image=is_type_input_image)

        if is_type_input_image:
            in_image = self._standardize(in_image)

        if is_reshape_input_image:
            in_image = np.squeeze(in_image, axis=-1)
        return in_image

    def _get_inverse_transformed_image(self, in_image: np.ndarray, is_type_input_image: bool = False) -> np.ndarray:
        if ImagesUtil.is_without_channels(self._size_image, in_image.shape):
            in_image = np.expand_dims(in_image, axis=-1)
            is_reshape_input_image = True
        else:
            is_reshape_input_image = False

        if is_type_input_image:
            in_image = self._standardize_inverse(in_image)

        in_image = self._calc_inverse_transformed_image(in_image, is_type_input_image=is_type_input_image)

        if is_reshape_input_image:
            in_image = np.squeeze(in_image, axis=-1)
        return in_image

    def _calc_transformed_image(self, in_array: np.ndarray, is_type_input_image: bool = False) -> np.ndarray:
        raise NotImplementedError

    def _calc_inverse_transformed_image(self, in_array: np.ndarray, is_type_input_image: bool = False) -> np.ndarray:
        raise NotImplementedError

    def _calc_gendata_random_transform(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError

    def _calc_gendata_inverse_random_transform(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError

    def _standardize(self, in_image: np.ndarray) -> np.ndarray:
        if self._preprocessing_function:
            in_image = self._preprocessing_function(in_image)
        if self._rescale_factor:
            in_image *= self._rescale_factor
        if self._samplewise_center:
            in_image -= np.mean(in_image, keepdims=True)
        if self._samplewise_std_normalization:
            in_image /= (np.std(in_image, keepdims=True) + _epsilon)

        template_message_error = 'This ImageDataGenerator specifies \'%s\', but it hasn\'t been fit on any ' \
                                 'training data. Fit it first by calling \'fit(numpy_data)\'.'

        if self._featurewise_center:
            if self._mean is not None:
                in_image -= self._mean
            else:
                message = template_message_error % ('featurewise_center')
                catch_error_exception(message)

        if self._featurewise_std_normalization:
            if self._std is not None:
                in_image /= (self._std + _epsilon)
            else:
                message = template_message_error % ('featurewise_std_normalization')
                catch_error_exception(template_message_error % (message))

        if self._is_zca_whitening:
            if self._principal_components is not None:
                flatx = np.reshape(in_image, (-1, np.prod(in_image.shape[-3:])))
                whitex = np.dot(flatx, self._principal_components)
                in_image = np.reshape(whitex, in_image.shape)
            else:
                message = template_message_error % ('zca_whitening')
                catch_error_exception(message)

        return in_image

    def _standardize_inverse(self, in_image: np.ndarray) -> np.ndarray:
        template_message_error = 'This ImageDataGenerator specifies \'%s\', but it hasn\'t been fit on any ' \
                                 'training data. Fit it first by calling \'fit(numpy_data)\'.'

        if self._is_zca_whitening:
            if self._principal_components is not None:
                flatx = np.reshape(in_image, (-1, np.prod(in_image.shape[-3:])))
                inverse_principal_componens = np.divide(1.0, self._principal_components)
                whitex = np.dot(flatx, inverse_principal_componens)
                in_image = np.reshape(whitex, in_image.shape)
            else:
                message = template_message_error % ('zca_whitening')
                catch_error_exception(message)

        if self._featurewise_std_normalization:
            if self._std is not None:
                in_image *= self._std
            else:
                message = template_message_error % ('featurewise_std_normalization')
                catch_error_exception(message)

        if self._featurewise_center:
            if self._mean is not None:
                in_image += self._mean
            else:
                message = template_message_error % ('featurewise_center')
                catch_error_exception(message)

        if self._samplewise_std_normalization:
            in_image *= np.std(in_image, keepdims=True)
        if self._samplewise_center:
            in_image += np.mean(in_image, keepdims=True)
        if self._rescale_factor:
            in_image /= self._rescale_factor
        if self._preprocessing_function:
            catch_error_exception('Not implemented inverse preprocessing function')

        return in_image

    @staticmethod
    def _flip_axis(in_image: np.ndarray, axis: int) -> np.ndarray:
        in_image = np.asarray(in_image).swapaxes(axis, 0)
        in_image = in_image[::-1, ...]
        in_image = in_image.swapaxes(0, axis)
        return in_image

    @staticmethod
    def _apply_channel_shift(in_image: np.ndarray, intensity: int, channel_axis: int = 0) -> np.ndarray:
        in_image = np.rollaxis(in_image, channel_axis, 0)
        min_x, max_x = np.min(in_image), np.max(in_image)
        channel_images = [np.clip(x_channel + intensity, min_x, max_x) for x_channel in in_image]
        in_image = np.stack(channel_images, axis=0)
        in_image = np.rollaxis(in_image, 0, channel_axis + 1)
        return in_image

    def _apply_brightness_shift(self, in_image: np.ndarray, brightness: int) -> np.ndarray:
        catch_error_exception('Not implemented brightness shifting option...')
        # in_image = array_to_img(in_image)
        # in_image = imgenhancer_Brightness = ImageEnhance.Brightness(in_image)
        # in_image = imgenhancer_Brightness.enhance(brightness)
        # in_image = img_to_array(in_image)

    def get_text_description(self) -> str:
        raise NotImplementedError


class TransformRigidImages2D(TransformRigidImages):
    _img_row_axis = 0
    _img_col_axis = 1
    _img_channel_axis = 2

    def __init__(self,
                 size_image: Tuple[int, int],
                 is_normalize_data: bool = False,
                 type_normalize_data: str = 'samplewise',
                 is_zca_whitening: bool = False,
                 rotation_range: float = 0.0,
                 width_shift_range: float = 0.0,
                 height_shift_range: float = 0.0,
                 brightness_range: Tuple[float, float] = None,
                 shear_range: float = 0.0,
                 zoom_range: Union[float, Tuple[float, float]] = 0.0,
                 channel_shift_range: float = 0.0,
                 fill_mode: str = 'nearest',
                 cval: float = 0.0,
                 horizontal_flip: bool = False,
                 vertical_flip: bool = False,
                 rescale_factor: float = None,
                 preprocessing_function: Callable[[np.ndarray], np.ndarray] = None
                 ) -> None:
        self._rotation_range = rotation_range
        self._width_shift_range = width_shift_range
        self._height_shift_range = height_shift_range
        self._brightness_range = brightness_range
        self._shear_range = shear_range
        self._channel_shift_range = channel_shift_range
        self._fill_mode = fill_mode
        self._cval = cval
        self._horizontal_flip = horizontal_flip
        self._vertical_flip = vertical_flip

        if np.isscalar(zoom_range):
            self._zoom_range = (1 - zoom_range, 1 + zoom_range)
        elif len(zoom_range) == 2:
            self._zoom_range = (zoom_range[0], zoom_range[1])
        else:
            message = '\'zoom_range\' should be a float or a tuple of two floats. Received %s' % (str(zoom_range))
            catch_error_exception(message)

        if self._brightness_range is not None:
            if len(self._brightness_range) != 2:
                message = '\'brightness_range\' should be a tuple of two floats. Received %s' % (str(brightness_range))
                catch_error_exception(message)

        super(TransformRigidImages2D, self).__init__(size_image,
                                                     is_normalize_data=is_normalize_data,
                                                     type_normalize_data=type_normalize_data,
                                                     is_zca_whitening=is_zca_whitening,
                                                     rescale_factor=rescale_factor,
                                                     preprocessing_function=preprocessing_function)

    def _calc_transformed_image(self, in_image: np.ndarray, is_type_input_image: bool = False) -> np.ndarray:
        # Apply: 1st: rigid transformations
        #        2nd: channel shift intensity / flipping
        if self._transform_matrix is not None:
            in_image = self._apply_transform(in_image, self._transform_matrix,
                                             channel_axis=self._img_channel_axis,
                                             fill_mode=self._fill_mode, cval=self._cval)

        if is_type_input_image and (self._transform_params.get('channel_shift_intensity') is not None):
            in_image = self._apply_channel_shift(in_image, self._transform_params['channel_shift_intensity'],
                                                 channel_axis=self._img_channel_axis)

        if self._transform_params.get('flip_horizontal', False):
            in_image = self._flip_axis(in_image, axis=self._img_col_axis)

        if self._transform_params.get('flip_vertical', False):
            in_image = self._flip_axis(in_image, axis=self._img_row_axis)

        if is_type_input_image and (self._transform_params.get('brightness') is not None):
            in_image = self._apply_brightness_shift(in_image, self._transform_params['brightness'])

        return in_image

    def _calc_inverse_transformed_image(self, in_image: np.ndarray, is_type_input_image: bool = False) -> np.ndarray:
        # Apply: 1st: channel shift intensity / flipping
        #        2nd: rigid transformations
        if is_type_input_image and (self._transform_params.get('brightness') is not None):
            in_image = self._apply_brightness_shift(in_image, self._transform_params['brightness'])

        if self._transform_params.get('flip_vertical', False):
            in_image = self._flip_axis(in_image, axis=self._img_row_axis)

        if self._transform_params.get('flip_horizontal', False):
            in_image = self._flip_axis(in_image, axis=self._img_col_axis)

        if is_type_input_image and (self._transform_params.get('channel_shift_intensity') is not None):
            in_image = self._apply_channel_shift(in_image, self._transform_params['channel_shift_intensity'],
                                                 channel_axis=self._img_channel_axis)

        if self._transform_matrix is not None:
            in_image = self._apply_transform(in_image, self._transform_matrix,
                                             channel_axis=self._img_channel_axis,
                                             fill_mode=self._fill_mode, cval=self._cval)
        return in_image

    def _calc_gendata_random_transform(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # compute composition of homographies
        if seed is not None:
            np.random.seed(seed)

        # ****************************************************
        if self._rotation_range:
            theta = np.deg2rad(np.random.uniform(-self._rotation_range, self._rotation_range))
        else:
            theta = 0

        if self._height_shift_range:
            tx = np.random.uniform(-self._height_shift_range, self._height_shift_range)
            if np.max(self._height_shift_range) < 1:
                tx *= self._size_image[self._img_row_axis]
        else:
            tx = 0

        if self._width_shift_range:
            ty = np.random.uniform(-self._width_shift_range, self._width_shift_range)
            if np.max(self._width_shift_range) < 1:
                ty *= self._size_image[self._img_col_axis]
        else:
            ty = 0

        if self._shear_range:
            shear = np.deg2rad(np.random.uniform(-self._shear_range, self._shear_range))
        else:
            shear = 0

        if self._zoom_range[0] == 1 and self._zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self._zoom_range[0], self._zoom_range[1], 2)

        flip_horizontal = (np.random.random() < 0.5) * self._horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self._vertical_flip

        channel_shift_intensity = None
        if self._channel_shift_range != 0:
            channel_shift_intensity = np.random.uniform(-self._channel_shift_range, self._channel_shift_range)

        brightness = None
        if self._brightness_range is not None:
            brightness = np.random.uniform(self._brightness_range[0], self._brightness_range[1])

        transform_parameters = {'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                'channel_shift_intensity': channel_shift_intensity,
                                'brightness': brightness}
        # ****************************************************

        # ****************************************************
        transform_matrix = None

        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = self._size_image[self._img_row_axis], self._size_image[self._img_col_axis]
            transform_matrix = self._transform_matrix_offset_center(transform_matrix, h, w)
        # ****************************************************

        return (transform_matrix, transform_parameters)

    def _calc_gendata_inverse_random_transform(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # compute composition of inverse homographies
        if seed is not None:
            np.random.seed(seed)

        # ****************************************************
        if self._rotation_range:
            theta = np.deg2rad(np.random.uniform(-self._rotation_range, self._rotation_range))
        else:
            theta = 0

        if self._height_shift_range:
            tx = np.random.uniform(-self._height_shift_range, self._height_shift_range)
            if self._height_shift_range < 1:
                tx *= self._size_image[self._img_row_axis]
        else:
            tx = 0
        if self._width_shift_range:
            ty = np.random.uniform(-self._width_shift_range, self._width_shift_range)
            if self._width_shift_range < 1:
                ty *= self._size_image[self._img_col_axis]
        else:
            ty = 0

        if self._shear_range:
            shear = np.deg2rad(np.random.uniform(-self._shear_range, self._shear_range))
        else:
            shear = 0

        if self._zoom_range[0] == 1 and self._zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self._zoom_range[0], self._zoom_range[1], 2)

        flip_horizontal = (np.random.random() < 0.5) * self._horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self._vertical_flip

        channel_shift_intensity = None
        if self._channel_shift_range != 0:
            channel_shift_intensity = np.random.uniform(-self._channel_shift_range, self._channel_shift_range)

        brightness = None
        if self._brightness_range is not None:
            brightness = np.random.uniform(self._brightness_range[0], self._brightness_range[1])

        transform_parameters = {'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                'channel_shift_intensity': channel_shift_intensity,
                                'brightness': brightness}
        # ****************************************************

        # ****************************************************
        transform_matrix = None

        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), np.sin(theta), 0],
                                        [-np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, -tx],
                                     [0, 1, -ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, np.tan(shear), 0],
                                     [0, 1.0 / np.cos(shear), 0],
                                     [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[1.0 / zx, 0, 0],
                                    [0, 1.0 / zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = self._size_image[self._img_row_axis], self._size_image[self._img_col_axis]
            transform_matrix = self._transform_matrix_offset_center(transform_matrix, h, w)
        # ****************************************************

        return (transform_matrix, transform_parameters)

    @staticmethod
    def _transform_matrix_offset_center(matrix: np.ndarray, x: int, y: int) -> np.ndarray:
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    @staticmethod
    def _apply_transform(in_image: np.ndarray, transform_matrix: np.ndarray,
                         channel_axis: int = 0, fill_mode: str = 'nearest', cval: float = 0.0) -> np.ndarray:
        in_image = np.rollaxis(in_image, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix, final_offset, order=1,
                                                             mode=fill_mode, cval=cval) for x_channel in in_image]
        in_image = np.stack(channel_images, axis=0)
        in_image = np.rollaxis(in_image, 0, channel_axis + 1)
        return in_image

    def get_text_description(self) -> str:
        message = 'Rigid 2D transformations of images, with parameters...\n'
        message += 'rotation (plane_XY) range: \'%s\'...\n' % (self._rotation_range)
        message += 'shift (width, height) range: \'(%s, %s)\'...\n' \
                   % (self._width_shift_range, self._height_shift_range)
        message += 'flip (horizontal, vertical): \'(%s, %s)\'...\n' \
                   % (self._horizontal_flip, self._vertical_flip)
        message += 'zoom (min, max) range: \'(%s, %s)\'...\n' % (self._zoom_range[0], self._zoom_range[1])
        message += 'shear (plane_XY) range: \'%s\'...\n' % (self._shear_range)
        message += 'fill mode, when applied transformation: \'%s\'...\n' % (self._fill_mode)
        return message


class TransformRigidImages3D(TransformRigidImages):
    _img_dep_axis = 0
    _img_row_axis = 1
    _img_col_axis = 2
    _img_channel_axis = 3

    def __init__(self,
                 size_image: Tuple[int, int, int],
                 is_normalize_data: bool = False,
                 type_normalize_data: str = 'samplewise',
                 is_zca_whitening: bool = False,
                 rotation_xy_range: float = 0.0,
                 rotation_xz_range: float = 0.0,
                 rotation_yz_range: float = 0.0,
                 width_shift_range: float = 0.0,
                 height_shift_range: float = 0.0,
                 depth_shift_range: float = 0.0,
                 brightness_range: Tuple[float, float] = None,
                 shear_xy_range: float = 0.0,
                 shear_xz_range: float = 0.0,
                 shear_yz_range: float = 0.0,
                 zoom_range: Union[float, Tuple[float, float]] = 0.0,
                 channel_shift_range: float = 0.0,
                 fill_mode: str = 'nearest',
                 cval: float = 0.0,
                 horizontal_flip: bool = False,
                 vertical_flip: bool = False,
                 axialdir_flip: bool = False,
                 rescale_factor: float = None,
                 preprocessing_function: Callable[[np.ndarray], np.ndarray] = None
                 ) -> None:
        self._rotation_xy_range = rotation_xy_range
        self._rotation_xz_range = rotation_xz_range
        self._rotation_yz_range = rotation_yz_range
        self._width_shift_range = width_shift_range
        self._height_shift_range = height_shift_range
        self._depth_shift_range = depth_shift_range
        self._brightness_range = brightness_range
        self._shear_xy_range = shear_xy_range
        self._shear_xz_range = shear_xz_range
        self._shear_yz_range = shear_yz_range
        self._channel_shift_range = channel_shift_range
        self._fill_mode = fill_mode
        self._cval = cval
        self._horizontal_flip = horizontal_flip
        self._vertical_flip = vertical_flip
        self._axialdir_flip = axialdir_flip

        if np.isscalar(zoom_range):
            self._zoom_range = (1 - zoom_range, 1 + zoom_range)
        elif len(zoom_range) == 2:
            self._zoom_range = (zoom_range[0], zoom_range[1])
        else:
            message = '\'zoom_range\' should be a float or a tuple of two floats. Received %s' % (str(zoom_range))
            catch_error_exception(message)

        if self._brightness_range is not None:
            if len(self._brightness_range) != 2:
                message = '\'brightness_range\' should be a tuple of two floats. Received %s' % (str(brightness_range))
                catch_error_exception(message)

        super(TransformRigidImages3D, self).__init__(size_image,
                                                     is_normalize_data=is_normalize_data,
                                                     type_normalize_data=type_normalize_data,
                                                     is_zca_whitening=is_zca_whitening,
                                                     rescale_factor=rescale_factor,
                                                     preprocessing_function=preprocessing_function)

    def _calc_transformed_image(self, in_image: np.ndarray, is_type_input_image: bool = False) -> np.ndarray:
        # Apply: 1st: rigid transformations
        #        2nd: channel shift intensity / flipping
        if self._transform_matrix is not None:
            in_image = self._apply_transform(in_image, self._transform_matrix,
                                             channel_axis=self._img_channel_axis,
                                             fill_mode=self._fill_mode, cval=self._cval)

        if is_type_input_image and (self._transform_params.get('channel_shift_intensity') is not None):
            in_image = self._apply_channel_shift(in_image, self._transform_params['channel_shift_intensity'],
                                                 channel_axis=self._img_channel_axis)

        if self._transform_params.get('flip_horizontal', False):
            in_image = self._flip_axis(in_image, axis=self._img_col_axis)

        if self._transform_params.get('flip_vertical', False):
            in_image = self._flip_axis(in_image, axis=self._img_row_axis)

        if self._transform_params.get('flip_axialdir', False):
            in_image = self._flip_axis(in_image, axis=self._img_dep_axis)

        if is_type_input_image and (self._transform_params.get('brightness') is not None):
            in_image = self._apply_brightness_shift(in_image, self._transform_params['brightness'])

        return in_image

    def _calc_inverse_transformed_image(self, in_image: np.ndarray, is_type_input_image: bool = False) -> np.ndarray:
        # Apply: 1st: channel shift intensity / flipping
        #        2nd: rigid transformations
        if is_type_input_image and (self._transform_params.get('brightness') is not None):
            in_image = self._apply_brightness_shift(in_image, self._transform_params['brightness'])

        if self._transform_params.get('flip_axialdir', False):
            in_image = self._flip_axis(in_image, axis=self._img_dep_axis)

        if self._transform_params.get('flip_vertical', False):
            in_image = self._flip_axis(in_image, axis=self._img_row_axis)

        if self._transform_params.get('flip_horizontal', False):
            in_image = self._flip_axis(in_image, axis=self._img_col_axis)

        if is_type_input_image and (self._transform_params.get('channel_shift_intensity') is not None):
            in_image = self._apply_channel_shift(in_image, self._transform_params['channel_shift_intensity'],
                                                 channel_axis=self._img_channel_axis)

        if self._transform_matrix is not None:
            in_image = self._apply_transform(in_image, self._transform_matrix,
                                             channel_axis=self._img_channel_axis,
                                             fill_mode=self._fill_mode, cval=self._cval)
        return in_image

    def _calc_gendata_random_transform(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # compute composition of homographies
        if seed is not None:
            np.random.seed(seed)

        # ****************************************************
        if self._rotation_xy_range:
            angle_xy = np.deg2rad(np.random.uniform(-self._rotation_xy_range, self._rotation_xy_range))
        else:
            angle_xy = 0
        if self._rotation_xz_range:
            angle_xz = np.deg2rad(np.random.uniform(-self._rotation_xz_range, self._rotation_xz_range))
        else:
            angle_xz = 0
        if self._rotation_yz_range:
            angle_yz = np.deg2rad(np.random.uniform(-self._rotation_yz_range, self._rotation_yz_range))
        else:
            angle_yz = 0

        if self._height_shift_range:
            tx = np.random.uniform(-self._height_shift_range, self._height_shift_range)
            if self._height_shift_range < 1:
                tx *= self._size_image[self._img_row_axis]
        else:
            tx = 0
        if self._width_shift_range:
            ty = np.random.uniform(-self._width_shift_range, self._width_shift_range)
            if self._width_shift_range < 1:
                ty *= self._size_image[self._img_col_axis]
        else:
            ty = 0
        if self._depth_shift_range:
            tz = np.random.uniform(-self._depth_shift_range, self._depth_shift_range)
            if self._depth_shift_range < 1:
                tz *= self._size_image[self._img_dep_axis]
        else:
            tz = 0

        if self._shear_xy_range:
            shear_xy = np.deg2rad(np.random.uniform(-self._shear_xy_range, self._shear_xy_range))
        else:
            shear_xy = 0
        if self._shear_xz_range:
            shear_xz = np.deg2rad(np.random.uniform(-self._shear_xz_range, self._shear_xz_range))
        else:
            shear_xz = 0
        if self._shear_yz_range:
            shear_yz = np.deg2rad(np.random.uniform(-self._shear_yz_range, self._shear_yz_range))
        else:
            shear_yz = 0

        if self._zoom_range[0] == 1 and self._zoom_range[1] == 1:
            (zx, zy, zz) = (1, 1, 1)
        else:
            (zx, zy, zz) = np.random.uniform(self._zoom_range[0], self._zoom_range[1], 3)

        flip_horizontal = (np.random.random() < 0.5) * self._horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self._vertical_flip
        flip_axialdir = (np.random.random() < 0.5) * self._axialdir_flip

        channel_shift_intensity = None
        if self._channel_shift_range != 0:
            channel_shift_intensity = np.random.uniform(-self._channel_shift_range, self._channel_shift_range)

        brightness = None
        if self._brightness_range is not None:
            brightness = np.random.uniform(self._brightness_range[0], self._brightness_range[1])

        transform_parameters = {'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                'flip_axialdir': flip_axialdir,
                                'channel_shift_intensity': channel_shift_intensity,
                                'brightness': brightness}
        # ****************************************************

        # ****************************************************
        transform_matrix = None

        if angle_xy != 0:
            rotation_matrix = np.array([[1, 0, 0, 0],
                                        [0, np.cos(angle_xy), -np.sin(angle_xy), 0],
                                        [0, np.sin(angle_xy), np.cos(angle_xy), 0],
                                        [0, 0, 0, 1]])
            transform_matrix = rotation_matrix
        if angle_xz != 0:
            rotation_matrix = np.array([[np.cos(angle_xz), np.sin(angle_xz), 0, 0],
                                        [-np.sin(angle_xz), np.cos(angle_xz), 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
            transform_matrix = \
                rotation_matrix if transform_matrix is None else np.dot(transform_matrix, rotation_matrix)
        if angle_yz != 0:
            rotation_matrix = np.array([[np.cos(angle_yz), 0, np.sin(angle_yz), 0],
                                        [0, 1, 0, 0],
                                        [-np.sin(angle_yz), 0, np.cos(angle_yz), 0],
                                        [0, 0, 0, 1]])
            transform_matrix = \
                rotation_matrix if transform_matrix is None else np.dot(transform_matrix, rotation_matrix)

        if tx != 0 or ty != 0 or tz != 0:
            shift_matrix = np.array([[1, 0, 0, tz],
                                     [0, 1, 0, tx],
                                     [0, 0, 1, ty],
                                     [0, 0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear_xy != 0:
            shear_matrix = np.array([[1, 0, 0, 0],
                                     [0, 1, -np.sin(shear_xy), 0],
                                     [0, 0, np.cos(shear_xy), 0],
                                     [0, 0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
        if shear_xz != 0:
            shear_matrix = np.array([[np.cos(shear_xz), 0, 0, 0],
                                     [-np.sin(shear_xz), 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
        if shear_yz != 0:
            shear_matrix = np.array([[np.cos(shear_yz), 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [-np.sin(shear_yz), 0, 1, 0],
                                     [0, 0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1 or zz != 1:
            zoom_matrix = np.array([[zz, 0, 0, 0],
                                    [0, zx, 0, 0],
                                    [0, 0, zy, 0],
                                    [0, 0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            (d, h, w) = (self._size_image[self._img_dep_axis],
                         self._size_image[self._img_row_axis],
                         self._size_image[self._img_col_axis])
            transform_matrix = self._transform_matrix_offset_center(transform_matrix, d, h, w)
        # ****************************************************

        return (transform_matrix, transform_parameters)

    def _calc_gendata_inverse_random_transform(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # compute composition of inverse homographies
        if seed is not None:
            np.random.seed(seed)

        # ****************************************************
        if self._rotation_xy_range:
            angle_xy = np.deg2rad(np.random.uniform(-self._rotation_xy_range, self._rotation_xy_range))
        else:
            angle_xy = 0
        if self._rotation_xz_range:
            angle_xz = np.deg2rad(np.random.uniform(-self._rotation_xz_range, self._rotation_xz_range))
        else:
            angle_xz = 0
        if self._rotation_yz_range:
            angle_yz = np.deg2rad(np.random.uniform(-self._rotation_yz_range, self._rotation_yz_range))
        else:
            angle_yz = 0

        if self._height_shift_range:
            tx = np.random.uniform(-self._height_shift_range, self._height_shift_range)
            if self._height_shift_range < 1:
                tx *= self._size_image[self._img_row_axis]
        else:
            tx = 0
        if self._width_shift_range:
            ty = np.random.uniform(-self._width_shift_range, self._width_shift_range)
            if self._width_shift_range < 1:
                ty *= self._size_image[self._img_col_axis]
        else:
            ty = 0
        if self._depth_shift_range:
            tz = np.random.uniform(-self._depth_shift_range, self._depth_shift_range)
            if self._depth_shift_range < 1:
                tz *= self._size_image[self._img_dep_axis]
        else:
            tz = 0

        if self._shear_xy_range:
            shear_xy = np.deg2rad(np.random.uniform(-self._shear_xy_range, self._shear_xy_range))
        else:
            shear_xy = 0
        if self._shear_xz_range:
            shear_xz = np.deg2rad(np.random.uniform(-self._shear_xz_range, self._shear_xz_range))
        else:
            shear_xz = 0
        if self._shear_yz_range:
            shear_yz = np.deg2rad(np.random.uniform(-self._shear_yz_range, self._shear_yz_range))
        else:
            shear_yz = 0

        if self._zoom_range[0] == 1 and self._zoom_range[1] == 1:
            (zx, zy, zz) = (1, 1, 1)
        else:
            (zx, zy, zz) = np.random.uniform(self._zoom_range[0], self._zoom_range[1], 3)

        flip_horizontal = (np.random.random() < 0.5) * self._horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self._vertical_flip
        flip_axialdir = (np.random.random() < 0.5) * self._axialdir_flip

        channel_shift_intensity = None
        if self._channel_shift_range != 0:
            channel_shift_intensity = np.random.uniform(-self._channel_shift_range, self._channel_shift_range)

        brightness = None
        if self._brightness_range is not None:
            brightness = np.random.uniform(self._brightness_range[0], self._brightness_range[1])

        transform_parameters = {'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                'flip_axialdir': flip_axialdir,
                                'channel_shift_intensity': channel_shift_intensity,
                                'brightness': brightness}
        # ****************************************************

        # ****************************************************
        transform_matrix = None

        if angle_xy != 0:
            rotation_matrix = np.array([[1, 0, 0, 0],
                                        [0, np.cos(angle_xy), np.sin(angle_xy), 0],
                                        [0, -np.sin(angle_xy), np.cos(angle_xy), 0],
                                        [0, 0, 0, 1]])
            transform_matrix = rotation_matrix
        if angle_xz != 0:
            rotation_matrix = np.array([[np.cos(angle_xz), -np.sin(angle_xz), 0, 0],
                                        [np.sin(angle_xz), np.cos(angle_xz), 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
            transform_matrix = \
                rotation_matrix if transform_matrix is None else np.dot(transform_matrix, rotation_matrix)
        if angle_yz != 0:
            rotation_matrix = np.array([[np.cos(angle_yz), 0, -np.sin(angle_yz), 0],
                                        [0, 1, 0, 0],
                                        [np.sin(angle_yz), 0, np.cos(angle_yz), 0],
                                        [0, 0, 0, 1]])
            transform_matrix = \
                rotation_matrix if transform_matrix is None else np.dot(transform_matrix, rotation_matrix)

        if tx != 0 or ty != 0 or tz != 0:
            shift_matrix = np.array([[1, 0, 0, -tz],
                                     [0, 1, 0, -tx],
                                     [0, 0, 1, -ty],
                                     [0, 0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear_xy != 0:
            shear_matrix = np.array([[1, 0, 0, 0],
                                     [0, 1, np.tan(shear_xy), 0],
                                     [0, 0, 1.0 / np.cos(shear_xy), 0],
                                     [0, 0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
        if shear_xz != 0:
            shear_matrix = np.array([[1.0 / np.cos(shear_xz), 0, 0, 0],
                                     [np.tan(shear_xz), 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
        if shear_yz != 0:
            shear_matrix = np.array([[1.0 / np.cos(shear_yz), 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [np.tan(shear_yz), 0, 1, 0],
                                     [0, 0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1 or zz != 1:
            zoom_matrix = np.array([[1.0 / zz, 0, 0, 0],
                                    [0, 1.0 / zx, 0, 0],
                                    [0, 0, 1.0 / zy, 0],
                                    [0, 0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            (d, h, w) = (self._size_image[self._img_dep_axis],
                         self._size_image[self._img_row_axis],
                         self._size_image[self._img_col_axis])
            transform_matrix = self._transform_matrix_offset_center(transform_matrix, d, h, w)
        # ****************************************************

        return (transform_matrix, transform_parameters)

    @staticmethod
    def _transform_matrix_offset_center(matrix: np.ndarray, x: int, y: int, z: int) -> np.ndarray:
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        o_z = float(z) / 2 + 0.5
        offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0, 0, 1, o_z], [0, 0, 0, 1]])
        reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0, 0, 1, -o_z], [0, 0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    @staticmethod
    def _apply_transform(in_image: np.ndarray, transform_matrix: np.ndarray,
                         channel_axis: int = 0, fill_mode: str = 'nearest', cval: float = 0.0) -> np.ndarray:
        in_image = np.rollaxis(in_image, channel_axis, 0)
        final_affine_matrix = transform_matrix[:3, :3]
        final_offset = transform_matrix[:3, 3]
        channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix, final_offset, order=0,
                                                             mode=fill_mode, cval=cval) for x_channel in in_image]
        in_image = np.stack(channel_images, axis=0)
        in_image = np.rollaxis(in_image, 0, channel_axis + 1)
        return in_image

    def get_text_description(self) -> str:
        message = 'Rigid 3D transformations of images, with parameters...\n'
        message += '- rotation (plane_XY, plane_XZ, plane_YZ) range: \'(%s, %s, %s)\'...\n' \
                   % (self._rotation_xy_range, self._rotation_xz_range, self._rotation_yz_range)
        message += '- shift (width, height, depth) range: \'(%s, %s, %s)\'...\n' \
                   % (self._width_shift_range, self._height_shift_range, self._depth_shift_range)
        message += '- flip (horizontal, vertical, axialdir): \'(%s, %s, %s)\'...\n' \
                   % (self._horizontal_flip, self._vertical_flip, self._axialdir_flip)
        message += '- zoom (min, max) range: \'(%s, %s)\'...\n' % (self._zoom_range[0], self._zoom_range[1])
        message += '- shear (plane_XY, plane_XZ, plane_YZ) range: \'(%s, %s, %s)\'...\n' \
                   % (self._shear_xy_range, self._shear_xz_range, self._shear_yz_range)
        message += '- fill mode, when applied transformation: \'%s\'...\n' % (self._fill_mode)
        return message
