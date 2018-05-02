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
from keras.preprocessing import image
import scipy.ndimage as ndi
import numpy as np
np.random.seed(2017)


class TransformationImages(object):

    def __init__(self, size_image):
        self.size_image = size_image

    def complete_init_data(self, size_total):
        pass

    def get_num_images(self):
        return 1

    def get_num_channels_array(self, in_array_shape):
        if len(in_array_shape) == len(self.size_image):
            return 1
        else:
            return in_array_shape[-1]

    def get_shape_out_array(self, in_array_shape):
        num_images   = in_array_shape[0]
        num_channels = self.get_num_channels_array(in_array_shape[1:])

        return [num_images] + list(self.size_image) + [num_channels]

    def get_transformed_image(self, images_array, seed=None):
        pass

    def get_image_array(self, images_array, seed=None):

        if len(images_array.shape) == len(self.size_image):
            return self.get_transformed_image(np.expand_dims(images_array, axis=-1), seed=seed)
        else:
            return self.get_transformed_image(images_array, seed=seed)

    def get_images_array_all(self, images_array, seed=None, diff_trans_batch=True):

        out_images_array = np.ndarray(self.get_shape_out_array(images_array.shape), dtype=images_array.dtype)

        num_images = images_array.shape[0]
        for index in range(num_images):
            if diff_trans_batch:
                seed_image = (seed if seed else 0) + index
            else:
                seed_image = seed
            out_images_array[index] = self.get_image_array(images_array[index], seed=seed_image)
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
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 fill_mode='nearest',
                 cval=0.0):

        if is_normalize_data:
            if type_normalize_data == 'featurewise':
                featurewise_center = True
                featurewise_std_normalization = True
                samplewise_center = False
                samplewise_std_normalization = False
            else: #type_normalize_data == 'samplewise'
                featurewise_center = False
                featurewise_std_normalization = False
                samplewise_center = True
                samplewise_std_normalization = True
        else:
            featurewise_center = False
            featurewise_std_normalization = False
            samplewise_center = False
            samplewise_std_normalization = False

        zca_epsilon = 1e-6

        self.images_data_generator = image.ImageDataGenerator(featurewise_center=featurewise_center,
                                                              samplewise_center=samplewise_center,
                                                              featurewise_std_normalization=featurewise_std_normalization,
                                                              samplewise_std_normalization=samplewise_std_normalization,
                                                              zca_whitening=zca_whitening,
                                                              zca_epsilon=zca_epsilon,
                                                              rotation_range=rotation_range,
                                                              width_shift_range=width_shift_range,
                                                              height_shift_range=height_shift_range,
                                                              shear_range=shear_range,
                                                              zoom_range=zoom_range,
                                                              channel_shift_range=channel_shift_range,
                                                              horizontal_flip=horizontal_flip,
                                                              vertical_flip=vertical_flip,
                                                              rescale=rescale,
                                                              fill_mode=fill_mode,
                                                              cval=cval,
                                                              data_format='channels_last')

        super(TransformationImages2D, self).__init__(size_image)

    def get_transformed_image(self, images_array, seed):

        return self.images_data_generator.standardize(self.images_data_generator.random_transform(images_array, seed=seed))


class TransformationImages3D(TransformationImages):

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
                 horizontal_flip=False,
                 vertical_flip=False,
                 depthZ_flip=False,
                 rescale=None,
                 fill_mode='nearest',
                 cval=0.0):

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
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.depthZ_flip = depthZ_flip
        self.rescale = rescale
        self.fill_mode = fill_mode
        self.cval = cval

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(self.zoom_range):
            self.zoom_range = (1 + self.zoom_range, 1 + self.zoom_range, 1 + self.zoom_range)
        elif len(self.zoom_range) != 3:
            CatchErrorException('`zoom_range` should be a float or a tuple or list of two floats. Received arg: ', self.zoom_range)

        if self.zca_whitening:
            if not self.featurewise_center:
                self.featurewise_center = True
                CatchWarningException('This ImageDataGenerator specifies `zca_whitening`, which overrides setting of `featurewise_center`.')
            if self.featurewise_std_normalization:
                self.featurewise_std_normalization = False
                CatchWarningException('This ImageDataGenerator specifies `zca_whitening`, which overrides setting of `featurewise_std_normalization`.')

        if self.featurewise_std_normalization:
            if not self.featurewise_center:
                self.featurewise_center = True
                CatchWarningException('This ImageDataGenerator specifies `featurewise_std_normalization`, which overrides setting of `featurewise_center`.')

        if self.samplewise_std_normalization:
            if not self.samplewise_center:
                self.samplewise_center = True
                CatchWarningException('This ImageDataGenerator specifies `samplewise_std_normalization`, which overrides setting of `samplewise_center`.')

        super(TransformationImages3D, self).__init__(size_image)

    def get_transformed_image(self, images_array, seed):

        return self.standardize(self.random_transform(images_array, seed=seed))


    def standardize(self, x):

        if self.rescale:
            x *= self.rescale
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, keepdims=True) + 1e-7)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                CatchWarningException('This ImageDataGenerator specifies `featurewise_center`, but it hasn\'t been fit on any training data. Fit it first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                CatchWarningException('This ImageDataGenerator specifies `featurewise_std_normalization`, but it hasn\'t been fit on any training data. Fit it first by calling `.fit(numpy_data)`.')

        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                CatchWarningException('This ImageDataGenerator specifies `zca_whitening`, but it hasn\'t been fit on any training data. Fit it first by calling `.fit(numpy_data)`.')

        return x


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
                tx *= x.shape[1]
        else:
            tx = 0
        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            if self.width_shift_range < 1:
                ty *= x.shape[2]
        else:
            ty = 0
        if self.depth_shift_range:
            tz = np.random.uniform(-self.depth_shift_range, self.depth_shift_range)
            if self.depth_shift_range < 1:
                tz *= x.shape[0]
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

        # Compute Rotation Matrix
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

        # Compute Translation Matrix
        if tx != 0 or ty != 0 or tz != 0:
            shift_matrix = np.array([[1, 0, 0, tz],
                                     [0, 1, 0, tx],
                                     [0, 0, 1, ty],
                                     [0, 0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        # Compute Shearing Matrix
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

        # Compute Zooming Matrix
        if zx != 1 or zy != 1 or zz != 1:
            zoom_matrix = np.array([[zz, 0, 0, 0],
                                    [0, zx, 0, 0],
                                    [0, 0, zy, 0],
                                    [0, 0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        # Apply transformations
        if transform_matrix is not None:
            (d, h, w) = x.shape[0:3]
            transform_matrix = self.transform_matrix_offset_center(transform_matrix, d, h, w)
            x = self.apply_transform(x, transform_matrix, channel_axis=3, fill_mode=self.fill_mode, cval=self.cval)


        if self.channel_shift_range != 0:
            x = image.random_channel_shift(x, self.channel_shift_range, channel_axis=-1)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = image.flip_axis(x, axis=2)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = image.flip_axis(x, axis=1)

        if self.depthZ_flip:
            if np.random.random() < 0.5:
                x = image.flip_axis(x, axis=0)

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