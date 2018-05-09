#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.ErrorMessages import *
from Preprocessing.BaseImageGenerator import *
from keras.preprocessing import image
import scipy.ndimage as ndi
import numpy as np
np.random.seed(2017)


class TransformationImages(BaseImageGenerator):

    fixed_seed_0_class     = 2017
    diff_trans_each_batch  = True
    is_update_fixed_seed_0 = True

    def __init__(self, size_image):
        self.size_image   = size_image
        self.fixed_seed_0 = self.fixed_seed_0_class

        super(TransformationImages, self).__init__(size_image)


    def complete_init_data(self, size_total):
        if self.is_update_fixed_seed_0:
            self.update_fixed_seed_0()

    def initialize_fixed_seed_0(self):
        self.fixed_seed_0 = self.fixed_seed_0_class

    def update_fixed_seed_0(self, new_fixed_seed_0=None):
        if new_fixed_seed_0:
            self.fixed_seed_0 = new_fixed_seed_0
        else:
            self.fixed_seed_0 = np.random.randint(2017)

    def get_mod_seed(self, seed):
        # update "seed" so that transformations are not fixed by the index of image_batch in a batch generator
        # "fixed_seed_0" should be changed after every loop over batches indexes
        if seed:
            return seed + self.fixed_seed_0
        else:
            return seed

    def get_seed_index_image(self, index, seed_0):
        if seed_0 and self.diff_trans_each_batch:
            return seed_0 + index
        else:
            return seed_0

    def get_num_images(self):
        return 1

    def get_shape_out_array(self, in_array_shape):
        num_images   = in_array_shape[0]
        num_channels = self.get_num_channels_array(in_array_shape[1:0])
        if num_channels:
            return [num_images] + list(self.size_image) + [num_channels]
        else:
            return [num_images] + list(self.size_image)


    def get_transformed_image(self, images_array, seed=None):
        pass

    def get_inverse_transformed_image(self, images_array, seed=None):
        pass

    def get_image_array(self, images_array, seed=None):

        if self.is_images_array_without_channels(images_array.shape):
            return np.squeeze(self.get_transformed_image(np.expand_dims(images_array, axis=-1), seed=self.get_mod_seed(seed)), axis=-1)
        else:
            return self.get_transformed_image(images_array, seed=self.get_mod_seed(seed))

    def get_inverse_image_array(self, images_array, seed=None):

        if self.is_images_array_without_channels(images_array.shape):
            return np.squeeze(self.get_inverse_transformed_image(np.expand_dims(images_array, axis=-1), seed=self.get_mod_seed(seed)), axis=-1)
        else:
            return self.get_inverse_transformed_image(images_array, seed=self.get_mod_seed(seed))

    def compute_images_array_all(self, images_array, seed_0=None):

        out_array_shape  = self.get_shape_out_array(images_array.shape)
        out_images_array = np.ndarray(out_array_shape, dtype=images_array.dtype)

        num_images = images_array.shape[0]
        for index in range(num_images):
            out_images_array[index] = self.get_image_array(images_array[index], seed=self.get_seed_index_image(index, seed_0))
        #endfor

        return out_images_array


class TransformationImages2D(TransformationImages):

    def __init__(self, size_image,
                 is_normalize_data=False,
                 type_normalize_data='samplewise',
                 zca_whitening=False,
                 rotation_range=0.0,
                 width_shift_range=0.0,
                 height_shift_range=0.0,
                 shear_range=0.0,
                 zoom_range=0.0,
                 channel_shift_range=0.0,
                 fill_mode='nearest',
                 cval=0.0,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None):

        if is_normalize_data:
            if type_normalize_data == 'featurewise':
                self.featurewise_center = True
                self.featurewise_std_normalization = True
                self.samplewise_center = False
                self.samplewise_std_normalization = False
            else: #type_normalize_data == 'samplewise'
                self.featurewise_center = False
                self.featurewise_std_normalization = False
                self.samplewise_center = True
                self.samplewise_std_normalization = True
        else:
            self.featurewise_center = False
            self.featurewise_std_normalization = False
            self.samplewise_center = False
            self.samplewise_std_normalization = False

        self.zca_whitening = zca_whitening
        self.zca_epsilon = 1e-6
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function

        self.mean = None
        self.std = None
        self.principal_components = None

        self.img_row_axis = 0
        self.img_col_axis = 1
        self.img_channel_axis = 2

        self.images_data_generator = image.ImageDataGenerator(featurewise_center=self.featurewise_center,
                                                              samplewise_center=self.samplewise_center,
                                                              featurewise_std_normalization=self.featurewise_std_normalization,
                                                              samplewise_std_normalization=self.samplewise_std_normalization,
                                                              zca_whitening=self.zca_whitening,
                                                              zca_epsilon=self.zca_epsilon,
                                                              rotation_range=self.rotation_range,
                                                              width_shift_range=self.width_shift_range,
                                                              height_shift_range=self.height_shift_range,
                                                              shear_range=self.shear_range,
                                                              zoom_range=self.zoom_range,
                                                              channel_shift_range=self.channel_shift_range,
                                                              fill_mode=self.fill_mode,
                                                              cval=self.cval,
                                                              horizontal_flip=self.horizontal_flip,
                                                              vertical_flip=self.vertical_flip,
                                                              rescale=self.rescale,
                                                              preprocessing_function=self.preprocessing_function,
                                                              data_format='channels_last')

        super(TransformationImages2D, self).__init__(size_image)


    def get_transformed_image(self, images_array, seed=None):

        return self.standardize(self.random_transform(images_array, seed=seed))

    def get_inverse_transformed_image(self, images_array, seed=None):

        return self.inverse_random_transform(self.inverse_standardize(images_array), seed=seed)


    def standardize(self, x):

        return self.images_data_generator.standardize(x)

    def inverse_standardize(self, x):

        self.mean                = self.images_data_generator.mean
        self.std                 = self.images_data_generator.std
        self.principal_components= self.images_data_generator.principal_components

        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                inverse_principal_componens = np.divide(1.0, self.principal_components)
                whitex = np.dot(flatx, inverse_principal_componens)
                x = np.reshape(whitex, x.shape)
            else:
                CatchErrorException('This ImageDataGenerator specifies '
                                    '`zca_whitening`, but it hasn\'t '
                                    'been fit on any training data. Fit it '
                                    'first by calling `.fit(numpy_data)`.')

        if self.featurewise_std_normalization:
            if self.std is not None:
                x *= self.std
            else:
                CatchErrorException('This ImageDataGenerator specifies '
                                    '`featurewise_std_normalization`, but it hasn\'t '
                                    'been fit on any training data. Fit it '
                                    'first by calling `.fit(numpy_data)`.')
        if self.featurewise_center:
            if self.mean is not None:
                x += self.mean
            else:
                CatchErrorException('This ImageDataGenerator specifies '
                                    '`featurewise_center`, but it hasn\'t '
                                    'been fit on any training data. Fit it '
                                    'first by calling `.fit(numpy_data)`.')

        if self.samplewise_std_normalization:
            x *= np.std(x, keepdims=True)
        if self.samplewise_center:
            x += np.mean(x, keepdims=True)
        if self.rescale:
            x /= self.rescale
        if self.preprocessing_function:
            CatchErrorException('Not implemented inverse preprocessing function')
            #x = self.preprocessing_function(x)

        return x


    def random_transform(self, x, seed=None):

        return self.images_data_generator.random_transform(x, seed=seed)

    def inverse_random_transform(self, x, seed=None):
        # Use composition of inverse transformations,
        # Apply progressively transforms in inverse order to that in "random_transform"
        # Useful when doing data augmentation during testing

        if seed is not None:
            np.random.seed(seed)

        if self.rotation_range:
            theta = np.deg2rad(np.random.uniform(-self.rotation_range, self.rotation_range))
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            if self.height_shift_range < 1:
                tx *= x.shape[self.img_row_axis]
        else:
            tx = 0
        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            if self.width_shift_range < 1:
                ty *= x.shape[self.img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.deg2rad(np.random.uniform(-self.shear_range, self.shear_range))
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)


        # Compute inverse transformations matrixes
        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[ np.cos(theta), np.sin(theta), 0],
                                        [-np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, -tx],
                                     [0, 1, -ty],
                                     [0, 0,  1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, np.tan(shear), 0],
                                     [0, 1.0/np.cos(shear), 0],
                                     [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[1.0/zx, 0, 0],
                                    [0, 1.0/zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)


        # Apply transformations (in inverse order)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = image.flip_axis(x, self.img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = image.flip_axis(x, self.img_row_axis)

        if self.channel_shift_range != 0:
            x = image.random_channel_shift(x, self.channel_shift_range, channel_axis=self.img_channel_axis)

        if transform_matrix is not None:
            h, w = x.shape[self.img_row_axis], x.shape[self.img_col_axis]
            transform_matrix = image.transform_matrix_offset_center(transform_matrix, h, w)
            x = image.apply_transform(x, transform_matrix,
                                      channel_axis=self.img_channel_axis,
                                      fill_mode=self.fill_mode,
                                      cval=self.cval)
        return x


class TransformationImages3D(TransformationImages2D):

    def __init__(self, size_image,
                 is_normalize_data=False,
                 type_normalize_data='samplewise',
                 zca_whitening=False,
                 rotation_XY_range=0.0,
                 rotation_XZ_range=0.0,
                 rotation_YZ_range=0.0,
                 width_shift_range=0.0,
                 height_shift_range=0.0,
                 depth_shift_range=0.0,
                 shear_XY_range=0.0,
                 shear_XZ_range=0.0,
                 shear_YZ_range=0.0,
                 zoom_range=0.0,
                 channel_shift_range=0.0,
                 fill_mode='nearest',
                 cval=0.0,
                 horizontal_flip=False,
                 vertical_flip=False,
                 depthZ_flip=False,
                 rescale=None,
                 preprocessing_function=None):

        self.rotation_XY_range = rotation_XY_range
        self.rotation_XZ_range = rotation_XZ_range
        self.rotation_YZ_range = rotation_YZ_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.depth_shift_range = depth_shift_range
        self.shear_XY_range = shear_XY_range
        self.shear_XZ_range = shear_XZ_range
        self.shear_YZ_range = shear_YZ_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.depthZ_flip = depthZ_flip

        # Delegate to "Transformation2D" the functions to "standardize" images
        # only provide parameters relevant for "standardize" functions
        super(TransformationImages3D, self).__init__(size_image,
                                                     is_normalize_data=is_normalize_data,
                                                     type_normalize_data=type_normalize_data,
                                                     zca_whitening=zca_whitening,
                                                     rescale=rescale,
                                                     preprocessing_function=preprocessing_function)
        self.img_dep_axis = 0
        self.img_row_axis = 1
        self.img_col_axis = 2
        self.img_channel_axis = 3

        if np.isscalar(self.zoom_range):
            self.zoom_range = (1 + self.zoom_range, 1 + self.zoom_range, 1 + self.zoom_range)
        elif len(self.zoom_range) != 3:
            CatchErrorException('`zoom_range` should be a float or a tuple or list of two floats. Received arg: ', self.zoom_range)


    def random_transform(self, x, seed=None):
        #Randomly augment a single image tensor.
        # use composition of homographies
        # to generate final transform that needs to be applied

        if seed is not None:
            np.random.seed(seed)

        if self.rotation_XY_range:
            angle_XY = np.deg2rad(np.random.uniform(-self.rotation_XY_range, self.rotation_XY_range))
        else:
            angle_XY = 0
        if self.rotation_XZ_range:
            angle_XZ = np.deg2rad(np.random.uniform(-self.rotation_XZ_range, self.rotation_XZ_range))
        else:
            angle_XZ = 0
        if self.rotation_YZ_range:
            angle_YZ = np.deg2rad(np.random.uniform(-self.rotation_YZ_range, self.rotation_YZ_range))
        else:
            angle_YZ = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            if self.height_shift_range < 1:
                tx *= x.shape[self.img_row_axis]
        else:
            tx = 0
        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            if self.width_shift_range < 1:
                ty *= x.shape[self.img_col_axis]
        else:
            ty = 0
        if self.depth_shift_range:
            tz = np.random.uniform(-self.depth_shift_range, self.depth_shift_range)
            if self.depth_shift_range < 1:
                tz *= x.shape[self.img_dep_axis]
        else:
            tz = 0

        if self.shear_XY_range:
            shear_XY = np.deg2rad(np.random.uniform(-self.shear_XY_range, self.shear_XY_range))
        else:
            shear_XY = 0
        if self.shear_XZ_range:
            shear_XZ = np.deg2rad(np.random.uniform(-self.shear_XZ_range, self.shear_XZ_range))
        else:
            shear_XZ = 0
        if self.shear_YZ_range:
            shear_YZ = np.deg2rad(np.random.uniform(-self.shear_YZ_range, self.shear_YZ_range))
        else:
            shear_YZ = 0

        if self.zoom_range == (1, 1, 1):
            (zx, zy, zz) = (1, 1, 1)
        else:
            (zx, zy, zz) = np.random.uniform(self.zoom_range[0], self.zoom_range[1], self.zoom_range[2], 3)


        transform_matrix = None
        if angle_XY != 0:
            rotation_matrix = np.array([[1, 0, 0, 0],
                                        [0, np.cos(angle_XY),-np.sin(angle_XY), 0],
                                        [0, np.sin(angle_XY), np.cos(angle_XY), 0],
                                        [0, 0, 0, 1]])
            transform_matrix = rotation_matrix
        if angle_XZ != 0:
            rotation_matrix = np.array([[ np.cos(angle_XZ), np.sin(angle_XZ), 0, 0],
                                        [-np.sin(angle_XZ), np.cos(angle_XZ), 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
            transform_matrix = rotation_matrix if transform_matrix is None else np.dot(transform_matrix, rotation_matrix)
        if angle_YZ != 0:
            rotation_matrix = np.array([[ np.cos(angle_YZ), 0, np.sin(angle_YZ), 0],
                                        [0, 1, 0, 0],
                                        [-np.sin(angle_YZ), 0, np.cos(angle_YZ), 0],
                                        [0, 0, 0, 1]])
            transform_matrix = rotation_matrix if transform_matrix is None else np.dot(transform_matrix, rotation_matrix)

        if tx != 0 or ty != 0 or tz != 0:
            shift_matrix = np.array([[1, 0, 0, tz],
                                     [0, 1, 0, tx],
                                     [0, 0, 1, ty],
                                     [0, 0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear_XY != 0:
            shear_matrix = np.array([[1, 0, 0, 0],
                                     [0, 1,-np.sin(shear_XY), 0],
                                     [0, 0, np.cos(shear_XY), 0],
                                     [0, 0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
        if shear_XZ != 0:
            shear_matrix = np.array([[ np.cos(shear_XZ), 0, 0, 0],
                                     [-np.sin(shear_XZ), 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
        if shear_YZ != 0:
            shear_matrix = np.array([[ np.cos(shear_YZ), 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [-np.sin(shear_YZ), 0, 1, 0],
                                     [0, 0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1 or zz != 1:
            zoom_matrix = np.array([[zz, 0, 0, 0],
                                    [0, zx, 0, 0],
                                    [0, 0, zy, 0],
                                    [0, 0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)


        # Apply transformations
        if transform_matrix is not None:
            (d, h, w) = (x.shape[self.img_dep_axis], x.shape[self.img_row_axis], x.shape[self.img_col_axis])
            transform_matrix = self.transform_matrix_offset_center(transform_matrix, d, h, w)
            x = self.apply_transform(x, transform_matrix,
                                     channel_axis=self.img_channel_axis,
                                     fill_mode=self.fill_mode,
                                     cval=self.cval)

        if self.channel_shift_range != 0:
            x = image.random_channel_shift(x, self.channel_shift_range, channel_axis=self.img_channel_axis)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = image.flip_axis(x, axis=self.img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = image.flip_axis(x, axis=self.img_row_axis)

        if self.depthZ_flip:
            if np.random.random() < 0.5:
                x = image.flip_axis(x, axis=self.img_dep_axis)

        return x


    def inverse_random_transform(self, x, seed=None):
        # Use composition of inverse transformations,
        # Apply progressively transforms in inverse order to that in "random_transform"
        # Useful when doing data augmentation during testing

        if seed is not None:
            np.random.seed(seed)

        if self.rotation_XY_range:
            angle_XY = np.deg2rad(np.random.uniform(-self.rotation_XY_range, self.rotation_XY_range))
        else:
            angle_XY = 0
        if self.rotation_XZ_range:
            angle_XZ = np.deg2rad(np.random.uniform(-self.rotation_XZ_range, self.rotation_XZ_range))
        else:
            angle_XZ = 0
        if self.rotation_YZ_range:
            angle_YZ = np.deg2rad(np.random.uniform(-self.rotation_YZ_range, self.rotation_YZ_range))
        else:
            angle_YZ = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            if self.height_shift_range < 1:
                tx *= x.shape[self.img_row_axis]
        else:
            tx = 0
        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            if self.width_shift_range < 1:
                ty *= x.shape[self.img_col_axis]
        else:
            ty = 0
        if self.depth_shift_range:
            tz = np.random.uniform(-self.depth_shift_range, self.depth_shift_range)
            if self.depth_shift_range < 1:
                tz *= x.shape[self.img_dep_axis]
        else:
            tz = 0

        if self.shear_XY_range:
            shear_XY = np.deg2rad(np.random.uniform(-self.shear_XY_range, self.shear_XY_range))
        else:
            shear_XY = 0
        if self.shear_XZ_range:
            shear_XZ = np.deg2rad(np.random.uniform(-self.shear_XZ_range, self.shear_XZ_range))
        else:
            shear_XZ = 0
        if self.shear_YZ_range:
            shear_YZ = np.deg2rad(np.random.uniform(-self.shear_YZ_range, self.shear_YZ_range))
        else:
            shear_YZ = 0

        if self.zoom_range == (1, 1, 1):
            (zx, zy, zz) = (1, 1, 1)
        else:
            (zx, zy, zz) = np.random.uniform(self.zoom_range[0], self.zoom_range[1], self.zoom_range[2], 3)


        # Compute inverse transformations matrixes
        transform_matrix = None
        if angle_XY != 0:
            rotation_matrix = np.array([[1, 0, 0, 0],
                                        [0, np.cos(angle_XY), np.sin(angle_XY), 0],
                                        [0,-np.sin(angle_XY), np.cos(angle_XY), 0],
                                        [0, 0, 0, 1]])
            transform_matrix = rotation_matrix
        if angle_XZ != 0:
            rotation_matrix = np.array([[np.cos(angle_XZ),-np.sin(angle_XZ), 0, 0],
                                        [np.sin(angle_XZ), np.cos(angle_XZ), 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
            transform_matrix = rotation_matrix if transform_matrix is None else np.dot(transform_matrix, rotation_matrix)
        if angle_YZ != 0:
            rotation_matrix = np.array([[np.cos(angle_YZ), 0,-np.sin(angle_YZ), 0],
                                        [0, 1, 0, 0],
                                        [np.sin(angle_YZ), 0, np.cos(angle_YZ), 0],
                                        [0, 0, 0, 1]])
            transform_matrix = rotation_matrix if transform_matrix is None else np.dot(transform_matrix, rotation_matrix)

        if tx != 0 or ty != 0 or tz != 0:
            shift_matrix = np.array([[1, 0, 0, -tz],
                                     [0, 1, 0, -tx],
                                     [0, 0, 1, -ty],
                                     [0, 0, 0,  1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear_XY != 0:
            shear_matrix = np.array([[1, 0, 0, 0],
                                     [0, 1, np.tan(shear_XY), 0],
                                     [0, 0, 1.0/np.cos(shear_XY), 0],
                                     [0, 0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
        if shear_XZ != 0:
            shear_matrix = np.array([[1.0/np.cos(shear_XZ), 0, 0, 0],
                                     [np.tan(shear_XZ), 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
        if shear_YZ != 0:
            shear_matrix = np.array([[1.0/np.cos(shear_YZ), 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [np.tan(shear_YZ), 0, 1, 0],
                                     [0, 0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1 or zz != 1:
            zoom_matrix = np.array([[1.0/zz, 0, 0, 0],
                                    [0, 1.0/zx, 0, 0],
                                    [0, 0, 1.0/zy, 0],
                                    [0, 0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)


        # Apply transformations (in inverse order)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = image.flip_axis(x, axis=self.img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = image.flip_axis(x, axis=self.img_row_axis)

        if self.depthZ_flip:
            if np.random.random() < 0.5:
                x = image.flip_axis(x, axis=self.img_dep_axis)

        if self.channel_shift_range != 0:
            x = image.random_channel_shift(x, self.channel_shift_range, channel_axis=self.img_channel_axis)

        if transform_matrix is not None:
            (d, h, w) = (x.shape[self.img_dep_axis], x.shape[self.img_row_axis], x.shape[self.img_col_axis])
            transform_matrix = self.transform_matrix_offset_center(transform_matrix, d, h, w)
            x = self.apply_transform(x, transform_matrix,
                                     channel_axis=self.img_channel_axis,
                                     fill_mode=self.fill_mode,
                                     cval=self.cval)
        return x


    @staticmethod
    def transform_matrix_offset_center(matrix, x, y, z):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        o_z = float(z) / 2 + 0.5
        offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0, 0, 1, o_z], [0, 0, 0, 1]])
        reset_matrix  = np.array([[1, 0, 0,-o_x], [0, 1, 0,-o_y], [0, 0, 1,-o_z], [0, 0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    @staticmethod
    def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:3, :3]
        final_offset = transform_matrix[:3, 3]
        channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix, final_offset,
                                                             order=0, mode=fill_mode, cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x