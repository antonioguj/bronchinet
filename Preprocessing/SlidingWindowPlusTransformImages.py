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

    def __init__(self, slidingWindow_generator, transformImages_generator):

        self.slidingWindow_generator   = slidingWindow_generator
        self.transformImages_generator = transformImages_generator

    def complete_init_data(self, size_total):
        self.slidingWindow_generator.complete_init_data(size_total)

    def get_num_images(self):
        return self.slidingWindow_generator.get_num_images()

    def get_shape_out_array(self, in_array_shape):

        return self.transformImages_generator.get_shape_out_array(self.slidingWindow_generator.get_shape_out_array(in_array_shape))

    def get_image_array(self, images_array, index, seed=None):

        return self.transformImages_generator.get_image_array(self.slidingWindow_generator.get_image_array(images_array, index), seed=seed)

    def get_images_array_all(self, images_array, seed=None, diff_trans_batch=True):

        out_images_array = np.ndarray(self.get_shape_out_array(images_array.shape), dtype=images_array.dtype)

        num_images = self.slidingWindow_generator.get_num_images()
        for index in range(num_images):
            if diff_trans_batch:
                seed_image = (seed if seed else 0) + index
            else:
                seed_image = seed
            out_images_array[index] = self.transformImages_generator.get_image_array(self.slidingWindow_generator.get_image_array(images_array, index), seed=seed_image)
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