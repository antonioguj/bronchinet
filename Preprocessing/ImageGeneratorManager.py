#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.Constants import *
from Preprocessing.BaseImageGenerator import *
from Preprocessing.RandomCropWindowImages import *
from Preprocessing.SlidingWindowImages import *
from Preprocessing.TransformationRigidImages import *
from Preprocessing.TransformElasticDeformImages import *



class CombinedImagesGenerator(BaseImageGenerator):

    def __init__(self, list_images_generators):
        self.list_images_generators = list_images_generators

        size_image = list_images_generators[-1].get_size_image()
        num_images = self.get_compute_num_images()

        super(CombinedImagesGenerator, self).__init__(size_image, num_images)


    #def complete_init_data(self, *args):
    def update_image_data(self, in_array_shape):
        for images_generator in self.list_images_generators:
            #images_generator.complete_init_data(args)
            images_generator.update_image_data(in_array_shape)
        #endfor
        self.num_images = self.get_compute_num_images()


    def compute_gendata(self, **kwargs):
        for images_generator in self.list_images_generators:
            images_generator.compute_gendata(**kwargs)
        #endfor

    def initialize_gendata(self):
        for images_generator in self.list_images_generators:
            images_generator.initialize_gendata()
        # endfor


    def get_text_description(self):
        message = ''
        for images_generator in self.list_images_generators:
            message += images_generator.get_text_description()
        #endfor
        return message


    def get_compute_num_images(self):
        num_images_prodrun = 1
        for images_generator in self.list_images_generators:
            num_images_prodrun *= images_generator.get_num_images()
        #endfor
        return num_images_prodrun


    def get_image(self, in_array):
        out_array = in_array
        for images_generator in self.list_images_generators:
            out_array = images_generator.get_image(out_array)
        #endfor
        return out_array



class NullGenerator(BaseImageGenerator):

    def __init__(self):
        super(NullGenerator, self).__init__(0, 0)

    #def complete_init_data(self, *args):
    def update_image_data(self, in_array_shape):
        pass

    def compute_gendata(self, **kwargs):
        return NotImplemented

    def initialize_gendata(self):
        return NotImplemented

    def get_text_description(self):
        return ''

    def get_image(self, in_array):
        return inout_array



def getImagesDataGenerator(size_in_images,
                           use_slidingWindowImages,
                           slidewindow_propOverlap,
                           use_randomCropWindowImages,
                           numRandomPatchesEpoch,
                           use_transformationRigidImages,
                           use_transformElasticDeformImages,
                           size_full_image= 0):

    list_images_generators = []

    if (use_slidingWindowImages):
        # generator of image patches by sliding-window...
        new_images_generator = SlidingWindowImages(size_in_images,
                                                   slidewindow_propOverlap,
                                                   size_full_image)
        list_images_generators.append(new_images_generator)

    elif (use_randomCropWindowImages):
        # generator of image patches by random cropping window...
        new_images_generator = RandomCropWindowImages(size_in_images,
                                                      numRandomPatchesEpoch,
                                                      size_full_image)
        list_images_generators.append(new_images_generator)


    if use_transformationRigidImages:
        # generator of images by random rigid transformations of input images...
        ndims = len(size_in_images)
        if ndims==2:
            new_images_generator = TransformationRigidImages2D(size_in_images,
                                                               rotation_range=ROTATION_XY_RANGE,
                                                               height_shift_range=HEIGHT_SHIFT_RANGE,
                                                               width_shift_range=WIDTH_SHIFT_RANGE,
                                                               horizontal_flip=HORIZONTAL_FLIP,
                                                               vertical_flip=VERTICAL_FLIP,
                                                               zoom_range=ZOOM_RANGE,
                                                               fill_mode=FILL_MODE_TRANSFORM)
            list_images_generators.append(new_images_generator)

        elif ndims==3:
            new_images_generator = TransformationRigidImages3D(size_in_images,
                                                               rotation_XY_range=ROTATION_XY_RANGE,
                                                               rotation_XZ_range=ROTATION_XZ_RANGE,
                                                               rotation_YZ_range=ROTATION_YZ_RANGE,
                                                               height_shift_range=HEIGHT_SHIFT_RANGE,
                                                               width_shift_range=WIDTH_SHIFT_RANGE,
                                                               depth_shift_range=DEPTH_SHIFT_RANGE,
                                                               horizontal_flip=HORIZONTAL_FLIP,
                                                               vertical_flip=VERTICAL_FLIP,
                                                               axialdir_flip=AXIALDIR_FLIP,
                                                               zoom_range=ZOOM_RANGE,
                                                               fill_mode=FILL_MODE_TRANSFORM)
            list_images_generators.append(new_images_generator)

        else:
            message = 'wrong value of \'ndims\' (%s)' %(ndims)
            CatchErrorException(message)


    if use_transformElasticDeformImages:
        if TYPETRANSFORMELASTICDEFORMATION == 'pixelwise':
            new_images_generator = TransformElasticDeformPixelwiseImages(size_in_images)
            list_images_generators.append(new_images_generator)

        else: #TYPETRANSFORMELASTICDEFORMATION == 'gridwise'
            new_images_generator = TransformElasticDeformGridwiseImages(size_in_images)
            list_images_generators.append(new_images_generator)


    num_created_images_generators = len(list_images_generators)

    if num_created_images_generators==0:
        return NullGenerator()

    elif num_created_images_generators==1:
        return list_images_generators[0]

    else: #num_created_images_generators>1:
        # generator of images as combination of single image generators
        return CombinedImagesGenerator(list_images_generators)