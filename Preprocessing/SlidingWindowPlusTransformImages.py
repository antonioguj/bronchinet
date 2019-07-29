#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Preprocessing.BaseImageGenerator import *
from Preprocessing.ElasticDeformationImages import *
from Preprocessing.SlidingWindowImages import *
from Preprocessing.TransformationImages import *



class SlidingWindowPlusTransformImages(BaseImageGenerator):

    def __init__(self, slidingWindow_generator,
                 transformImages_generator,
                 use_seed_transform= True):
        self.slidingWindow_generator = slidingWindow_generator
        self.transformImages_generator = transformImages_generator
        self.use_seed_transform = use_seed_transform

        super(SlidingWindowPlusTransformImages, self).__init__(slidingWindow_generator.size_image)


    def complete_init_data(self, in_array_shape):
        self.slidingWindow_generator.complete_init_data(in_array_shape)
        self.transformImages_generator.complete_init_data(in_array_shape)

    def get_num_images_dirs(self):
        return self.slidingWindow_generator.get_num_images_dirs()

    def get_limits_images_dirs(self):
        return self.slidingWindow_generator.get_limits_images_dirs()

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

    def get_images_array(self, images_array,
                         index,
                         masks_array= None,
                         seed= None):
        if masks_array is None:
            out_images_array = self.slidingWindow_generator.get_images_array(images_array, index)
            out_images_array = self.transformImages_generator.get_images_array(out_images_array, seed=seed)
            return out_images_array
        else:
            (out_images_array, out_masks_array) = self.slidingWindow_generator.get_images_array(images_array, index,
                                                                                                masks_array= masks_array)
            (out_images_array, out_masks_array) = self.transformImages_generator.get_images_array(out_images_array,
                                                                                                  masks_array= out_masks_array, seed=seed)
            return (out_images_array, out_masks_array)

    def compute_images_array_all(self, images_array,
                                 masks_array= None,
                                 seed_0= None):
        out_images_shape = self.get_shape_out_array(images_array.shape)
        out_images_array = np.ndarray(out_images_shape, dtype=images_array.dtype)

        if masks_array is None:
            num_images = self.get_num_images()
            for index in range(num_images):
                seed = self.get_seed_index_image(index, seed_0)
                out_images_array[index] = self.get_images_array(images_array, index, seed=seed)
            #endfor
            return out_images_array
        else:
            out_masks_shape = self.get_shape_out_array(masks_array.shape)
            out_masks_array = np.ndarray(out_masks_shape, dtype =masks_array.dtype)
            num_images = self.get_num_images()
            for index in range(num_images):
                seed = self.get_seed_index_image(index, seed_0)
                (out_images_array[index], out_masks_array[index]) = self.get_images_array(images_array, index,
                                                                                          masks_array= masks_array, seed=seed)
            #endfor
            return (out_images_array, out_masks_array)



class SlidingWindowPlusTransformImages2D(SlidingWindowPlusTransformImages):

    def __init__(self, size_image,
                 prop_overlap,
                 size_total=(0,0),
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
        super(SlidingWindowPlusTransformImages2D, self).__init__(SlidingWindowImages2D(size_image,
                                                                                       prop_overlap,
                                                                                       size_total=size_total),
                                                                 TransformationImages2D(size_image,
                                                                                        is_normalize_data=is_normalize_data,
                                                                                        type_normalize_data=type_normalize_data,
                                                                                        zca_whitening=zca_whitening,
                                                                                        rotation_range=rotation_range,
                                                                                        width_shift_range=width_shift_range,
                                                                                        height_shift_range=height_shift_range,
                                                                                        shear_range=shear_range,
                                                                                        zoom_range=zoom_range,
                                                                                        channel_shift_range=channel_shift_range,
                                                                                        fill_mode=fill_mode,
                                                                                        cval=cval,
                                                                                        horizontal_flip=horizontal_flip,
                                                                                        vertical_flip=vertical_flip,
                                                                                        rescale=rescale,
                                                                                        preprocessing_function=preprocessing_function))



class SlicingPlusTransformImages2D(SlidingWindowPlusTransformImages):

    def __init__(self, size_image,
                 size_total=(0,0),
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
        super(SlicingPlusTransformImages2D, self).__init__(SlicingImages2D(size_image,
                                                                           size_total=size_total),
                                                           TransformationImages2D(size_image,
                                                                                  is_normalize_data=is_normalize_data,
                                                                                  type_normalize_data=type_normalize_data,
                                                                                  zca_whitening=zca_whitening,
                                                                                  rotation_range=rotation_range,
                                                                                  width_shift_range=width_shift_range,
                                                                                  height_shift_range=height_shift_range,
                                                                                  shear_range=shear_range,
                                                                                  zoom_range=zoom_range,
                                                                                  channel_shift_range=channel_shift_range,
                                                                                  fill_mode=fill_mode,
                                                                                  cval=cval,
                                                                                  horizontal_flip=horizontal_flip,
                                                                                  vertical_flip=vertical_flip,
                                                                                  rescale=rescale,
                                                                                  preprocessing_function=preprocessing_function))

class SlidingWindowPlusElasticDeformationImages2D(SlidingWindowPlusTransformImages):

    def __init__(self, size_image,
                 prop_overlap,
                 size_total=(0,0,0),
                 type_elastic_deformation='gridwise'):
        if type_elastic_deformation == 'pixelwise':
            super(SlidingWindowPlusElasticDeformationImages2D, self).__init__(SlidingWindowImages2D(size_image,
                                                                                                    prop_overlap,
                                                                                                    size_total=size_total),
                                                                              ElasticDeformationPixelwiseImages2D(size_image))
        else: #type_elastic_deformation == 'gridwise'
            super(SlidingWindowPlusElasticDeformationImages2D, self).__init__(SlidingWindowImages2D(size_image,
                                                                                                    prop_overlap,
                                                                                                    size_total=size_total),
                                                                              ElasticDeformationGridwiseImages2D(size_image))

class SlicingPlusElasticDeformationImages2D(SlidingWindowPlusTransformImages):

    def __init__(self, size_image,
                 size_total=(0,0,0),
                 type_elastic_deformation='gridwise'):
        if type_elastic_deformation == 'pixelwise':
            super(SlicingPlusElasticDeformationImages2D, self).__init__(SlicingImages2D(size_image,
                                                                                        size_total=size_total),
                                                                        ElasticDeformationPixelwiseImages2D(size_image))
        else: #type_elastic_deformation == 'gridwise'
            super(SlicingPlusElasticDeformationImages2D, self).__init__(SlicingImages2D(size_image,
                                                                                        size_total=size_total),
                                                                        ElasticDeformationGridwiseImages2D(size_image))

class SlidingWindowPlusTransformImages3D(SlidingWindowPlusTransformImages):

    def __init__(self, size_image,
                 prop_overlap,
                 size_total=(0,0,0),
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
        super(SlidingWindowPlusTransformImages3D, self).__init__(SlidingWindowImages3D(size_image,
                                                                                       prop_overlap,
                                                                                       size_total=size_total),
                                                                 TransformationImages3D(size_image,
                                                                                        is_normalize_data=is_normalize_data,
                                                                                        type_normalize_data=type_normalize_data,
                                                                                        zca_whitening=zca_whitening,
                                                                                        rotation_XY_range=rotation_XY_range,
                                                                                        rotation_XZ_range=rotation_XZ_range,
                                                                                        rotation_YZ_range=rotation_YZ_range,
                                                                                        width_shift_range=width_shift_range,
                                                                                        height_shift_range=height_shift_range,
                                                                                        depth_shift_range=depth_shift_range,
                                                                                        shear_XY_range=shear_XY_range,
                                                                                        shear_XZ_range=shear_XZ_range,
                                                                                        shear_YZ_range=shear_YZ_range,
                                                                                        zoom_range=zoom_range,
                                                                                        channel_shift_range=channel_shift_range,
                                                                                        fill_mode=fill_mode,
                                                                                        cval=cval,
                                                                                        horizontal_flip=horizontal_flip,
                                                                                        vertical_flip=vertical_flip,
                                                                                        depthZ_flip=depthZ_flip,
                                                                                        rescale=rescale,
                                                                                        preprocessing_function=preprocessing_function))

class SlicingPlusTransformImages3D(SlidingWindowPlusTransformImages):

    def __init__(self, size_image,
                 size_total=(0,0,0),
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
        super(SlicingPlusTransformImages3D, self).__init__(SlicingImages3D(size_image,
                                                                           size_total=size_total),
                                                           TransformationImages3D(size_image,
                                                                                  is_normalize_data=is_normalize_data,
                                                                                  type_normalize_data=type_normalize_data,
                                                                                  zca_whitening=zca_whitening,
                                                                                  rotation_XY_range=rotation_XY_range,
                                                                                  rotation_XZ_range=rotation_XZ_range,
                                                                                  rotation_YZ_range=rotation_YZ_range,
                                                                                  width_shift_range=width_shift_range,
                                                                                  height_shift_range=height_shift_range,
                                                                                  depth_shift_range=depth_shift_range,
                                                                                  shear_XY_range=shear_XY_range,
                                                                                  shear_XZ_range=shear_XZ_range,
                                                                                  shear_YZ_range=shear_YZ_range,
                                                                                  zoom_range=zoom_range,
                                                                                  channel_shift_range=channel_shift_range,
                                                                                  fill_mode=fill_mode,
                                                                                  cval=cval,
                                                                                  horizontal_flip=horizontal_flip,
                                                                                  vertical_flip=vertical_flip,
                                                                                  depthZ_flip=depthZ_flip,
                                                                                  rescale=rescale,
                                                                                  preprocessing_function=preprocessing_function))

class SlidingWindowPlusElasticDeformationImages3D(SlidingWindowPlusTransformImages):

    def __init__(self, size_image,
                 prop_overlap,
                 size_total=(0,0,0),
                 type_elastic_deformation='gridwise'):
        if type_elastic_deformation == 'pixelwise':
            super(SlidingWindowPlusElasticDeformationImages3D, self).__init__(SlidingWindowImages3D(size_image,
                                                                                                    prop_overlap,
                                                                                                    size_total=size_total),
                                                                              ElasticDeformationPixelwiseImages3D(size_image))
        else: #type_elastic_deformation == 'gridwise'
            super(SlidingWindowPlusElasticDeformationImages3D, self).__init__(SlidingWindowImages3D(size_image,
                                                                                                    prop_overlap,
                                                                                                    size_total=size_total),
                                                                              ElasticDeformationGridwiseImages3D(size_image))

class SlicingPlusElasticDeformationImages3D(SlidingWindowPlusTransformImages):

    def __init__(self, size_image,
                 size_total=(0,0,0),
                 type_elastic_deformation='gridwise'):
        if type_elastic_deformation == 'pixelwise':
            super(SlicingPlusElasticDeformationImages3D, self).__init__(SlicingImages3D(size_image,
                                                                                        size_total=size_total),
                                                                        ElasticDeformationPixelwiseImages3D(size_image))
        else: #type_elastic_deformation == 'gridwise'
            super(SlicingPlusElasticDeformationImages3D, self).__init__(SlicingImages3D(size_image,
                                                                                        size_total=size_total),
                                                                        ElasticDeformationGridwiseImages3D(size_image))
