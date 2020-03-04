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
from Postprocessing.ImageReconstructor import *
from Preprocessing.FilteringUnetOutputValidConvs import *
from Preprocessing.ImageGeneratorManager import *



def getImagesReconstructor(size_in_images,
                           use_slidingWindowImages,
                           slidewindow_propOverlap,
                           use_randomCropWindowImages,
                           numRandomPatchesEpoch,
                           use_transformationRigidImages= False,
                           size_full_image= 0,
                           is_outputUnet_validconvs= False,
                           size_output_images= None,
                           is_filter_output_unet= False,
                           prop_filter_output_unet= None,
                           num_trans_per_sample= 1):

    images_generator = getImagesDataGenerator(size_in_images,
                                              use_slidingWindowImages,
                                              slidewindow_propOverlap,
                                              use_randomCropWindowImages,
                                              numRandomPatchesEpoch,
                                              use_transformationRigidImages,
                                              use_transformElasticDeformImages=False,
                                              size_full_image= size_full_image)

    if is_filter_output_unet:
        size_filter_output_unet = tuple([int(prop_filter_output_unet * elem) for elem in size_in_images])
        print("Filtering output probability maps of Unet, with a final output size: \'%s\'..." % (str(size_filter_output_unet)))

        ndims = len(size_in_images)
        if ndims==2:
            filter_images_generator = FilteringUnetOutputValidConvs2D(size_in_images, size_filter_output_unet)
        elif ndims==3:
            filter_images_generator = FilteringUnetOutputValidConvs3D(size_in_images, size_filter_output_unet)
        else:
            raise Exception('Error: self.ndims')
    else:
        filter_images_generator = None


    if not use_slidingWindowImages and not use_randomCropWindowImages:
        message = 'Image Reconstructor without sliding-window patches not implemented yet'
        CatchErrorException(message)


    if not use_transformationRigidImages:
        # reconstructor of images following the sliding-window generator of input patches
        images_reconstructor = ImageReconstructor(size_in_images,
                                                  images_generator,
                                                  size_full_image=size_full_image,
                                                  is_outputUnet_validconvs=is_outputUnet_validconvs,
                                                  size_output_image=size_output_images,
                                                  is_filter_output_unet=is_filter_output_unet,
                                                  filter_images_generator=filter_images_generator)
    else:
        # reconstructor of images accounting for transformations during testing...
        # IMPORTANT: PROTOTYPE, NOT TESTED YET
        images_reconstructor = ImageReconstructorWithTransformation(size_in_images,
                                                                    images_generator,
                                                                    num_trans_per_patch= num_trans_per_sample,
                                                                    size_full_image= size_full_image,
                                                                    is_outputUnet_validconvs= is_outputUnet_validconvs,
                                                                    size_output_image= size_output_images,
                                                                    is_filter_output_unet= is_filter_output_unet,
                                                                    filter_images_generator= filter_images_generator)

    return images_reconstructor