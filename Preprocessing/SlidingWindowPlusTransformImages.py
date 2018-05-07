#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Preprocessing.SlidingWindowImages import *
from Preprocessing.TransformationImages import *


class SlidingWindowPlusTransformImages(object):

    def __init__(self, slidingWindow_generator, transformImages_generator, use_seed_transform=True):

        self.slidingWindow_generator   = slidingWindow_generator
        self.transformImages_generator = transformImages_generator
        self.use_seed_transform        = use_seed_transform

    def complete_init_data(self, size_total):

        self.slidingWindow_generator  .complete_init_data(size_total)
        self.transformImages_generator.complete_init_data(size_total)

    def get_mod_seed(self, seed):
        if self.use_seed_transform:
            return self.transformImages_generator.get_mod_seed(seed)
        else:
            return False

    def get_seed_index_image(self, index, seed_0):
        return self.transformImages_generator.get_seed_index_image(index, seed_0)

    def get_num_images(self):
        return self.slidingWindow_generator.get_num_images()

    def get_shape_out_array(self, in_array_shape):
        return self.transformImages_generator.get_shape_out_array(self.slidingWindow_generator.get_shape_out_array(in_array_shape))

    def get_image_array(self, images_array, index):

        return self.transformImages_generator.get_image_array(self.slidingWindow_generator.get_image_array(images_array, index), seed=self.get_mod_seed(index))

    def compute_images_array_all(self, images_array, seed_0=None):
        if seed_0:
            self.transformImages_generator.update_fixed_seed_0(seed_0)

        out_array_shape  = self.get_shape_out_array(images_array.shape)
        out_images_array = np.ndarray(out_array_shape, dtype=images_array.dtype)

        num_images = self.get_num_images()
        for index in range(num_images):
            out_images_array[index] = self.get_image_array(images_array, index)
        #endfor

        return out_images_array


class SlidingWindowPlusTransformImages2D(SlidingWindowPlusTransformImages):

    def __init__(self,
                 size_image,
                 prop_overlap,
                 size_total=(0, 0),
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

        super(SlidingWindowPlusTransformImages2D, self).__init__(SlidingWindowImages2D(size_image,
                                                                                       prop_overlap,
                                                                                       size_total),
                                                                 TransformationImages2D(size_image,
                                                                                        is_normalize_data,
                                                                                        type_normalize_data,
                                                                                        zca_whitening,
                                                                                        rotation_range,
                                                                                        width_shift_range,
                                                                                        height_shift_range,
                                                                                        shear_range,
                                                                                        zoom_range,
                                                                                        channel_shift_range,
                                                                                        horizontal_flip,
                                                                                        vertical_flip,
                                                                                        rescale,
                                                                                        fill_mode,
                                                                                        cval))

class SlicingPlusTransformImages2D(SlidingWindowPlusTransformImages):

    def __init__(self,
                 size_image,
                 size_total=(0, 0),
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

        super(SlicingPlusTransformImages2D, self).__init__(SlicingImages2D(size_image,
                                                                           size_total),
                                                           TransformationImages2D(size_image,
                                                                                  is_normalize_data,
                                                                                  type_normalize_data,
                                                                                  zca_whitening,
                                                                                  rotation_range,
                                                                                  width_shift_range,
                                                                                  height_shift_range,
                                                                                  shear_range,
                                                                                  zoom_range,
                                                                                  channel_shift_range,
                                                                                  horizontal_flip,
                                                                                  vertical_flip,
                                                                                  rescale,
                                                                                  fill_mode,
                                                                                  cval))

class SlidingWindowPlusTransformImages3D(SlidingWindowPlusTransformImages):

    def __init__(self,
                 size_image,
                 prop_overlap,
                 size_total=(0, 0, 0),
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

        super(SlidingWindowPlusTransformImages3D, self).__init__(SlidingWindowImages3D(size_image,
                                                                                       prop_overlap,
                                                                                       size_total),
                                                                 TransformationImages3D(size_image,
                                                                                        is_normalize_data,
                                                                                        type_normalize_data,
                                                                                        zca_whitening,
                                                                                        rotation_XY_range,
                                                                                        rotation_XZ_range,
                                                                                        rotation_YZ_range,
                                                                                        width_shift_range,
                                                                                        height_shift_range,
                                                                                        depth_shift_range,
                                                                                        shear_XY_range,
                                                                                        shear_XZ_range,
                                                                                        shear_YZ_range,
                                                                                        zoom_range,
                                                                                        channel_shift_range,
                                                                                        horizontal_flip,
                                                                                        vertical_flip,
                                                                                        depthZ_flip,
                                                                                        rescale,
                                                                                        fill_mode,
                                                                                        cval))

class SlicingPlusTransformImages3D(SlidingWindowPlusTransformImages):

    def __init__(self,
                 size_image,
                 size_total=(0, 0, 0),
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

        super(SlicingPlusTransformImages3D, self).__init__(SlicingImages3D(size_image,
                                                                           size_total),
                                                           TransformationImages3D(size_image,
                                                                                  is_normalize_data,
                                                                                  type_normalize_data,
                                                                                  zca_whitening,
                                                                                  rotation_XY_range,
                                                                                  rotation_XZ_range,
                                                                                  rotation_YZ_range,
                                                                                  width_shift_range,
                                                                                  height_shift_range,
                                                                                  depth_shift_range,
                                                                                  shear_XY_range,
                                                                                  shear_XZ_range,
                                                                                  shear_YZ_range,
                                                                                  zoom_range,
                                                                                  channel_shift_range,
                                                                                  horizontal_flip,
                                                                                  vertical_flip,
                                                                                  depthZ_flip,
                                                                                  rescale,
                                                                                  fill_mode,
                                                                                  cval))