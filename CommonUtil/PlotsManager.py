#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.Constants import *
from CommonUtil.FileReaders import *
import matplotlib.pyplot as plt
import numpy as np
import os


class PlotsManager(object):

    counter = 0

    @staticmethod
    def plot_model_history(model_history):
        "Function to plot model accuracy and loss"

        fig, axs = plt.subplots(1,2,figsize=(15,5))
        # summarize history for accuracy
        axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
        axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
        axs[0].legend(['train', 'val'], loc='best')
        # summarize history for loss
        axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
        axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
        axs[1].legend(['train', 'val'], loc='best')
        plt.show()

    @staticmethod
    def plot_image_mask_2D(images, masks):

        num_images     = images.shape[0]
        num_plot_images= min(5, num_images)
        randIndexes = np.random.choice(num_images, size=num_plot_images, replace=False)

        fig, axis = plt.subplots(1, num_plot_images, figsize=(10,5))
        for i, index in enumerate(randIndexes):
            image_plot= images[index].reshape(IMAGES_HEIGHT, IMAGES_WIDTH)
            mask_plot = masks [index].reshape(IMAGES_HEIGHT, IMAGES_WIDTH)
            axis[i].set_title('Image %s' %(index))
            axis[i].imshow(image_plot)
            axis[i].imshow(mask_plot, alpha=0.9)
        plt.show()

    @classmethod
    def saveplot_image_mask_3D(cls, outfiles_path, images, masks):

        num_images     = images.shape[0]
        num_save_images= min(5, num_images)
        randIndexes = np.random.choice(num_images, size=num_save_images, replace=False)

        #save images in nifti format
        for i, index in enumerate(randIndexes):
            image_plot= images[index].reshape(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH)
            mask_plot = masks [index].reshape(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH)

            outname_images = os.path.join(outfiles_path, 'images-%0.2i.nii'%(cls.counter))
            outname_masks  = os.path.join(outfiles_path, 'masks-%0.2i.nii' %(cls.counter))
            NIFTIreader.writeImageArray(outname_images, np.swapaxes(image_plot, 0, 2))
            NIFTIreader.writeImageArray(outname_masks,  np.swapaxes(mask_plot,  0, 2))
            cls.counter += 1

    @staticmethod
    def plot_compare_images_2D(images, masks_predict, masks_truth):

        num_images     = images.shape[0]
        num_plot_images= min(5, num_images)
        randIndexes = np.random.choice(num_images, size=num_plot_images, replace=False)

        fig, axis = plt.subplots(2, num_plot_images, figsize=(10,5))
        for i, index in enumerate(randIndexes):
            image_plot       = images       [index].reshape(IMAGES_HEIGHT, IMAGES_WIDTH)
            mask_predict_plot= masks_predict[index].reshape(IMAGES_HEIGHT, IMAGES_WIDTH)
            mask_truth_plot  = masks_truth  [index].reshape(IMAGES_HEIGHT, IMAGES_WIDTH)
            axis[0, i].imshow(image_plot, cmap='gray')
            axis[0, i].imshow(mask_predict_plot, cmap='jet', alpha=0.5)
            axis[0, i].set_title('Predict %s' %(index))
            axis[1, i].imshow(image_plot, cmap='gray')
            axis[1, i].imshow(mask_truth_plot, cmap='jet', alpha=0.5)
            axis[1, i].set_title('Truth %s' %(index))
        plt.show()

    @staticmethod
    def plot_compare_images_3D(images, masks_predict, masks_truth):

        newShape = (images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])
        images        = images       .reshape(newShape)
        masks_predict = masks_predict.reshape(newShape)
        masks_truth   = masks_truth.  reshape(newShape)

        PlotsManager.plot_compare_images_2D(images, masks_predict, masks_truth)
