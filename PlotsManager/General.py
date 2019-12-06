#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.FunctionsUtil import *
import matplotlib.pyplot as plt
import numpy as np


class General(object):

    max_plot_image_array_axfigx = 3
    max_plot_image_array_axfigy = 5
    max_plot_image_array_figure = 15
    max_plot_image_array_axfigy_2 = 4
    max_plot_image_array_figure_2 = 8
    num_figures_saved = 16
    skip_slices_plot_image_array = 1


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


    @classmethod
    def plot_images_masks_randomSlices(cls, in_image_array, in_mask_array,
                                       num_plot_image_array= max_plot_image_array_figure,
                                       isave_outfiles= False,
                                       template_outfilename= 'slices_%s.png'):
        num_total_slices= in_image_array.shape[0]
        size_slices = (in_image_array.shape[1], in_image_array.shape[2])
        num_plot_image_array = min(num_plot_image_array, num_total_slices)
        indexes_slices = np.random.choice(num_total_slices, size=num_plot_image_array, replace=False)

        fig, axis = plt.subplots(cls.max_plot_image_array_axfigx, cls.max_plot_image_array_axfigy, figsize=(10,5))

        for i, index in enumerate(indexes_slices):
            (ind_i, ind_j) = (i//cls.max_plot_image_array_axfigy, i%cls.max_plot_image_array_axfigy)
            in_image_array_slice_plot= in_image_array[index].reshape(size_slices)
            in_mask_array_slice_plot = in_mask_array [index].reshape(size_slices)

            axis[ind_i, ind_j].set_title('slice %s' %(index))
            axis[ind_i, ind_j].imshow(in_image_array_slice_plot,cmap='gray')
            axis[ind_i, ind_j].imshow(in_mask_array_slice_plot, cmap='jet', alpha=0.5)
        #endfor

        if isave_outfiles:
            outfilename = template_outfilename %('_'.join([str(x) for x in indexes_slices]))
            plt.savefig(outfilename)
            plt.close()
        else:
            plt.show()


    @classmethod
    def plot_images_masks_allSlices(cls, in_image_array, in_mask_array,
                                    type_output= 2,
                                    isave_outfiles= False,
                                    template_outfilename= 'slices_%s.png'):
        num_total_slices = in_image_array.shape[0]
        size_slices = (in_image_array.shape[1], in_image_array.shape[2])

        if type_output==1:
            num_figures_saved = num_total_slices // (cls.skip_slices_plot_image_array * cls.max_plot_image_array_figure)
            range_slices_figure = cls.max_plot_image_array_figure * cls.skip_slices_plot_image_array
        elif type_output==2:
            num_figures_saved = cls.num_figures_saved
            range_slices_figure = num_total_slices // cls.num_figures_saved

        count_slice = 0
        for n in range(num_figures_saved):
            indexes_slices = np.linspace(count_slice, count_slice+range_slices_figure, num=cls.max_plot_image_array_figure+1, dtype=int)[:-1]

            fig, axis = plt.subplots(cls.max_plot_image_array_axfigx, cls.max_plot_image_array_axfigy, figsize=(10, 5))

            for i, index in enumerate(indexes_slices):
                (ind_i, ind_j) = (i//cls.max_plot_image_array_axfigy, i%cls.max_plot_image_array_axfigy)
                in_image_array_slice_plot = in_image_array[index].reshape(size_slices)
                in_mask_array_slice_plot  = in_mask_array [index].reshape(size_slices)

                axis[ind_i, ind_j].set_title('slice %s' % (index))
                axis[ind_i, ind_j].imshow(in_image_array_slice_plot, cmap='gray')
                axis[ind_i, ind_j].imshow(in_mask_array_slice_plot, cmap='jet', alpha=0.5)
            #endfor
            count_slice += range_slices_figure

            if isave_outfiles:
                outfilename = template_outfilename %('_'.join([str(x) for x in indexes_slices]))
                plt.savefig(outfilename)
                plt.close()
            else:
                plt.show()
        #endfor


    @classmethod
    def plot_compare_images_masks_allSlices(cls, in_image_array, orig_mask_array, predict_mask_array,
                                            type_output= 2,
                                            isave_outfiles= False,
                                            template_outfilename= 'slices_%s.png'):
        num_total_slices = in_image_array.shape[0]
        size_slices = (in_image_array.shape[1], in_image_array.shape[2])

        if type_output==1:
            num_figures_saved = num_total_slices // (cls.skip_slices_plot_image_array * cls.max_plot_image_array_axfigy_2)
            range_slices_figure = cls.max_plot_image_array_axfigy_2 * cls.skip_slices_plot_image_array
        elif type_output==2:
            num_figures_saved = cls.num_figures_saved
            range_slices_figure = num_total_slices // cls.num_figures_saved

        count_slice = 0
        for n in range(num_figures_saved):
            indexes_slices = np.linspace(count_slice, count_slice+range_slices_figure, num=cls.max_plot_image_array_axfigy_2+1, dtype=int)[:-1]

            fig, axis = plt.subplots(2, cls.max_plot_image_array_axfigy_2, figsize=(10,5))

            for i, index in enumerate(indexes_slices):
                in_image_array_slice_plot  = in_image_array [index].reshape(size_slices)
                orig_mask_array_slice_plot = orig_mask_array[index].reshape(size_slices)
                predict_mask_array_slice_plot = predict_mask_array[index].reshape(size_slices)

                axis[0, i].set_title('slice %s' % (index))
                axis[0, i].imshow(in_image_array_slice_plot, cmap='gray')
                axis[0, i].imshow(orig_mask_array_slice_plot, cmap='jet', alpha=0.5)
                axis[1, i].set_title('slice %s' % (index))
                axis[1, i].imshow(in_image_array_slice_plot, cmap='gray')
                axis[1, i].imshow(predict_mask_array_slice_plot, cmap='jet', alpha=0.5)
            #endfor
            count_slice += range_slices_figure

            if isave_outfiles:
                outfilename = template_outfilename %('_'.join([str(x) for x in indexes_slices]))
                plt.savefig(outfilename)
                plt.close()
            else:
                plt.show()
        #endfor


    @classmethod
    def plot_compare_transform_images_masks_allSlices(cls, orig_image_array, orig_mask_array,
                                                      transform_image_array, transform_mask_array,
                                                      type_output= 1,
                                                      isave_outfiles= False,
                                                      template_outfilename= 'slices_%s.png'):
        num_total_slices = orig_image_array.shape[0]
        size_slices = (orig_image_array.shape[1], orig_image_array.shape[2])

        if type_output==1:
            num_figures_saved = num_total_slices // (cls.skip_slices_plot_image_array * cls.max_plot_image_array_axfigy_2)
            range_slices_figure = cls.max_plot_image_array_axfigy_2 * cls.skip_slices_plot_image_array
        elif type_output==2:
            num_figures_saved = cls.num_figures_saved
            range_slices_figure = num_total_slices // cls.num_figures_saved

        count_slice = 0
        for n in range(num_figures_saved):
            indexes_slices = np.linspace(count_slice, count_slice+range_slices_figure, num=cls.max_plot_image_array_axfigy_2+1, dtype=int)[:-1]

            fig, axis = plt.subplots(2, cls.max_plot_image_array_axfigy_2, figsize=(10,5))

            for i, index in enumerate(indexes_slices):
                orig_image_array_slice_plot = orig_image_array[index].reshape(size_slices)
                orig_mask_array_slice_plot  = orig_mask_array [index].reshape(size_slices)
                transform_image_array_slice_plot = transform_image_array[index].reshape(size_slices)
                transform_mask_array_slice_plot  = transform_mask_array [index].reshape(size_slices)

                axis[0, i].set_title('slice %s' % (index))
                axis[0, i].imshow(orig_image_array_slice_plot, cmap='gray')
                axis[0, i].imshow(orig_mask_array_slice_plot, cmap='jet', alpha=0.5)
                axis[1, i].set_title('slice %s' % (index))
                axis[1, i].imshow(transform_image_array_slice_plot, cmap='gray')
                axis[1, i].imshow(transform_mask_array_slice_plot, cmap='jet', alpha=0.5)
            #endfor
            count_slice += range_slices_figure

            if isave_outfiles:
                outfilename = template_outfilename %('_'.join([str(x) for x in indexes_slices]))
                plt.savefig(outfilename)
                plt.close()
            else:
                plt.show()
        #endfor