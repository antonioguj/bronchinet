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
from CommonUtil.FunctionsUtil import *
import matplotlib.pyplot as plt
import numpy as np


class PlotsManager(object):

    max_plot_images_axfigx   = 3
    max_plot_images_axfigy   = 5
    max_plot_images_figure   = 15
    max_plot_images_axfigy_2 = 4
    max_plot_images_figure_2 = 8
    num_figures_saved        = 16
    skip_slices_plot_images  = 1


    def __init__(self, isave_images_files=False, out_files_pathname=None):
        self.isave_images_files = isave_images_files
        self.out_files_pathname = out_files_pathname
        if self.isave_images_files and \
            not isExistdir(out_files_pathname):
            message = "PlotsManager: output path '%s' does not exist..." %(out_files_pathname)
            CatchErrorException(message)


    def compute_plot_histogram(self, images_array, bins=20, name_file='image_test.png'):

        #(hist, bins) = np.histogram(images_array, bins=bins, range=[np.min(images_array), np.max(images_array)])
        plt.hist(images_array.flatten(), bins=bins, range=[np.min(images_array), np.max(images_array)], log=True)

        if self.isave_images_files:
            outfilename = joinpathnames(self.out_files_pathname, name_file)
            plt.savefig(outfilename)
            plt.close()
        else:
            plt.show()


    @staticmethod
    def plot_model_history(model_history):
        "function to plot model accuracy and loss"

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


    def plot_images_masks_randomSlices(self, images, masks, num_plot_images=max_plot_images_figure):

        num_total_slices= images.shape[0]
        size_slices     =(images.shape[1], images.shape[2])
        num_plot_images = min(num_plot_images, num_total_slices)
        indexes_slices  = np.random.choice(num_total_slices, size=num_plot_images, replace=False)

        fig, axis = plt.subplots(self.max_plot_images_axfigx, self.max_plot_images_axfigy, figsize=(10,5))

        for i, index in enumerate(indexes_slices):
            (ind_i, ind_j) = (i//self.max_plot_images_axfigy, i%self.max_plot_images_axfigy)

            images_slice_plot= images[index].reshape(size_slices)
            masks_slice_plot = masks [index].reshape(size_slices)

            axis[ind_i, ind_j].set_title('slice %s' %(index))
            axis[ind_i, ind_j].imshow(images_slice_plot,cmap='gray')
            axis[ind_i, ind_j].imshow(masks_slice_plot, cmap='jet', alpha=0.5)
        #endfor

        if self.isave_images_files:
            outfilename = 'slices_%s.png' %('_'.join([str(x) for x in indexes_slices]))
            outfilename = joinpathnames(self.out_files_pathname, outfilename)
            plt.savefig(outfilename)
            plt.close()
        else:
            plt.show()


    def plot_images_masks_allSlices(self, images, masks, typeOutput=2):

        num_total_slices= images.shape[0]
        size_slices     =(images.shape[1], images.shape[2])

        if typeOutput==1:
            num_figures_saved   = num_total_slices // (self.skip_slices_plot_images * self.max_plot_images_figure)
            range_slices_figure = self.max_plot_images_figure * self.skip_slices_plot_images
        elif typeOutput==2:
            num_figures_saved   = self.num_figures_saved
            range_slices_figure = num_total_slices // self.num_figures_saved

        count_slice = 0
        for n in range(num_figures_saved):
            fig, axis = plt.subplots(self.max_plot_images_axfigx, self.max_plot_images_axfigy, figsize=(10, 5))

            indexes_slices = np.linspace(count_slice, count_slice+range_slices_figure, num=self.max_plot_images_figure+1, dtype=int)[:-1]

            for i, index in enumerate(indexes_slices):
                (ind_i, ind_j) = (i//self.max_plot_images_axfigy, i%self.max_plot_images_axfigy)

                images_slice_plot = images[index].reshape(size_slices)
                masks_slice_plot  = masks [index].reshape(size_slices)

                axis[ind_i, ind_j].set_title('slice %s' % (index))
                axis[ind_i, ind_j].imshow(images_slice_plot, cmap='gray')
                axis[ind_i, ind_j].imshow(masks_slice_plot,  cmap='jet', alpha=0.5)
            #endfor

            count_slice += range_slices_figure

            if self.isave_images_files:
                outfilename = 'slices_%s.png' %('_'.join([str(x) for x in indexes_slices]))
                outfilename = joinpathnames(self.out_files_pathname, outfilename)
                plt.savefig(outfilename)
                plt.close()
            else:
                plt.show()
        #endfor


    def plot_compare_images_masks_allSlices(self, images, origin_masks, predict_masks, typeOutput=2):

        num_total_slices= images.shape[0]
        size_slices     =(images.shape[1], images.shape[2])

        if typeOutput==1:
            num_figures_saved   = num_total_slices // (self.skip_slices_plot_images * self.max_plot_images_axfigy_2)
            range_slices_figure = self.max_plot_images_axfigy_2 * self.skip_slices_plot_images
        elif typeOutput==2:
            num_figures_saved   = self.num_figures_saved
            range_slices_figure = num_total_slices // self.num_figures_saved

        count_slice = 0
        for n in range(num_figures_saved):
            fig, axis = plt.subplots(2, self.max_plot_images_axfigy_2, figsize=(10,5))

            indexes_slices = np.linspace(count_slice, count_slice+range_slices_figure, num=self.max_plot_images_axfigy_2+1, dtype=int)[:-1]

            for i, index in enumerate(indexes_slices):
                images_slice_plot        = images       [index].reshape(size_slices)
                origin_masks_slice_plot  = origin_masks [index].reshape(size_slices)
                predict_masks_slice_plot = predict_masks[index].reshape(size_slices)

                axis[0, i].set_title('slice %s' % (index))
                axis[0, i].imshow(images_slice_plot, cmap='gray')
                axis[0, i].imshow(origin_masks_slice_plot,  cmap='jet', alpha=0.5)
                axis[1, i].set_title('slice %s' % (index))
                axis[1, i].imshow(images_slice_plot, cmap='gray')
                axis[1, i].imshow(predict_masks_slice_plot, cmap='jet', alpha=0.5)
            #endfor

            count_slice += range_slices_figure

            if self.isave_images_files:
                outfilename = 'slices_%s.png' %('_'.join([str(x) for x in indexes_slices]))
                outfilename = joinpathnames(self.out_files_pathname, outfilename)
                plt.savefig(outfilename)
                plt.close()
            else:
                plt.show()
        #endfor


    def plot_compare_transform_images_masks_allSlices(self, origin_images, origin_masks, transform_images, transform_masks, typeOutput=1):

        num_total_slices= origin_images.shape[0]
        size_slices     =(origin_images.shape[1], origin_images.shape[2])

        if typeOutput==1:
            num_figures_saved   = num_total_slices // (self.skip_slices_plot_images * self.max_plot_images_axfigy_2)
            range_slices_figure = self.max_plot_images_axfigy_2 * self.skip_slices_plot_images
        elif typeOutput==2:
            num_figures_saved   = self.num_figures_saved
            range_slices_figure = num_total_slices // self.num_figures_saved

        count_slice = 0
        for n in range(num_figures_saved):
            fig, axis = plt.subplots(2, self.max_plot_images_axfigy_2, figsize=(10,5))

            indexes_slices = np.linspace(count_slice, count_slice+range_slices_figure, num=self.max_plot_images_axfigy_2+1, dtype=int)[:-1]

            for i, index in enumerate(indexes_slices):
                origin_images_slice_plot   = origin_images   [index].reshape(size_slices)
                origin_masks_slice_plot    = origin_masks    [index].reshape(size_slices)
                transform_images_slice_plot= transform_images[index].reshape(size_slices)
                transform_masks_slice_plot = transform_masks [index].reshape(size_slices)

                axis[0, i].set_title('slice %s' % (index))
                axis[0, i].imshow(origin_images_slice_plot, cmap='gray')
                axis[0, i].imshow(origin_masks_slice_plot,  cmap='jet', alpha=0.5)
                axis[1, i].set_title('slice %s' % (index))
                axis[1, i].imshow(transform_images_slice_plot, cmap='gray')
                axis[1, i].imshow(transform_masks_slice_plot,  cmap='jet', alpha=0.5)
            #endfor

            count_slice += range_slices_figure

            if self.isave_images_files:
                outfilename = 'slices_%s.png' %('_'.join([str(x) for x in indexes_slices]))
                outfilename = joinpathnames(self.out_files_pathname, outfilename)
                plt.savefig(outfilename)
                plt.close()
            else:
                plt.show()
        #endfor