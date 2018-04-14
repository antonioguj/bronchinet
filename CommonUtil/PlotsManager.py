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


class PlotsManager(object):

    max_plot_images_axfigx   = 3
    max_plot_images_axfigy   = 5
    max_plot_images_figure   = 15
    max_plot_images_axfigy_2 = 4
    max_plot_images_figure_2 = 8
    num_figures_saved        = 20
    skip_slices_plot_images  = 4


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


    @classmethod
    def plot_images_masks_randomSlices(cls, images, masks, num_plot_images=max_plot_images_figure,
                                       isSaveImages=False, outfilespath=None):

        num_total_slices= images.shape[0]
        size_slices     =(images.shape[1], images.shape[2])
        num_plot_images = min(num_plot_images, num_total_slices)
        indexes_slices  = np.random.choice(num_total_slices, size=num_plot_images, replace=False)

        fig, axis = plt.subplots(cls.max_plot_images_axfigx, cls.max_plot_images_axfigy, figsize=(10,5))

        for i, index in enumerate(indexes_slices):
            (ind_i, ind_j) = (i//cls.max_plot_images_axfigy, i%cls.max_plot_images_axfigy)

            images_slice_plot= images[index].reshape(size_slices)
            masks_slice_plot = masks [index].reshape(size_slices)

            axis[ind_i, ind_j].set_title('slice %s' %(index))
            axis[ind_i, ind_j].imshow(images_slice_plot,cmap='gray')
            axis[ind_i, ind_j].imshow(masks_slice_plot, cmap='jet', alpha=0.5)
        #endfor

        if isSaveImages:
            outfilename = joinpathnames(outfilespath, 'slices_%s.png' %('_'.join([str(x) for x in indexes_slices])))
            plt.savefig(outfilename)
        else:
            plt.show()


    @classmethod
    def plot_images_masks_allSlices(cls, images, masks, typeOutput=1, isSaveImages=False, outfilespath=None):

        num_total_slices= images.shape[0]
        size_slices     =(images.shape[1], images.shape[2])

        if typeOutput==1:
            num_figures_saved   = num_total_slices // (cls.skip_slices_plot_images * cls.max_plot_images_figure)
            range_slices_figure = cls.max_plot_images_figure * cls.skip_slices_plot_images
        elif typeOutput==2:
            num_figures_saved   = cls.num_figures_saved
            range_slices_figure = num_total_slices // cls.num_figures_saved

        count_slice = 0
        for n in range(num_figures_saved):
            fig, axis = plt.subplots(cls.max_plot_images_axfigx, cls.max_plot_images_axfigy, figsize=(10, 5))

            indexes_slices = np.linspace(count_slice, count_slice+range_slices_figure, num=cls.max_plot_images_figure+1, dtype=int)[:-1]

            for i, index in enumerate(indexes_slices):
                (ind_i, ind_j) = (i//cls.max_plot_images_axfigy, i%cls.max_plot_images_axfigy)

                images_slice_plot = images[index].reshape(size_slices)
                masks_slice_plot  = masks [index].reshape(size_slices)

                axis[ind_i, ind_j].set_title('slice %s' % (index))
                axis[ind_i, ind_j].imshow(images_slice_plot, cmap='gray')
                axis[ind_i, ind_j].imshow(masks_slice_plot,  cmap='jet', alpha=0.5)
            # endfor

            count_slice += range_slices_figure

            if isSaveImages:
                outfilename = joinpathnames(outfilespath, 'slices_%s.png' %('_'.join([str(x) for x in indexes_slices])))
                plt.savefig(outfilename)
                plt.close()
            else:
                plt.show()
        #endfor


    @classmethod
    def plot_compare_images_masks_allSlices(cls, images, origin_masks, predict_masks, typeOutput=1, isSaveImages=False, outfilespath=None):

        num_total_slices= images.shape[0]
        size_slices     =(images.shape[1], images.shape[2])

        if typeOutput==1:
            num_figures_saved   = num_total_slices // (cls.skip_slices_plot_images * cls.max_plot_images_axfigy_2)
            range_slices_figure = cls.max_plot_images_axfigy_2 * cls.skip_slices_plot_images
        elif typeOutput==2:
            num_figures_saved   = cls.num_figures_saved
            range_slices_figure = num_total_slices // cls.num_figures_saved

        count_slice = 0
        for n in range(num_figures_saved):
            fig, axis = plt.subplots(2, cls.max_plot_images_axfigy_2, figsize=(10,5))

            indexes_slices = np.linspace(count_slice, count_slice+range_slices_figure, num=cls.max_plot_images_axfigy_2+1, dtype=int)[:-1]

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
            # endfor

            count_slice += range_slices_figure

            if isSaveImages:
                outfilename = joinpathnames(outfilespath, 'slices_%s.png' %('_'.join([str(x) for x in indexes_slices])))
                plt.savefig(outfilename)
                plt.close()
            else:
                plt.show()
        #endfor