
import matplotlib.pyplot as plt
import numpy as np


class PlotGeneral(object):

    _max_plot_images_axfigx = 3
    _max_plot_images_axfigy = 5
    _max_plot_images_figure = 15
    _max_plot_images_axfigy_2 = 4
    _max_plot_images_figure_2 = 8
    _num_figures_saved = 16
    _skip_slices_plot_image = 1

    @staticmethod
    def plot_model_history(model_history) -> None:
        "function to plot model accuracy and loss"
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # summarize history for accuracy
        axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
        axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
        axs[0].legend(['train', 'val'], loc='best')

        # summarize history for loss
        axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
        axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
        axs[1].legend(['train', 'val'], loc='best')
        plt.show()

    @classmethod
    def plot_images_masks_random_slices(cls,
                                        in_image: np.ndarray,
                                        in_mask: np.ndarray,
                                        num_plot_images: int = _max_plot_images_figure,
                                        is_save_outfiles: bool = False,
                                        template_outfilename: str = 'slices_%s.png'
                                        ) -> None:
        num_total_slices = in_image.shape[0]
        size_slices = (in_image.shape[1], in_image.shape[2])
        num_plot_images = min(num_plot_images, num_total_slices)
        indexes_slices = np.random.choice(num_total_slices, size=num_plot_images, replace=False)

        fig, axis = plt.subplots(cls._max_plot_images_axfigx, cls._max_plot_images_axfigy, figsize=(10, 5))

        for i, index in enumerate(indexes_slices):
            (ind_i, ind_j) = (i // cls._max_plot_images_axfigy, i % cls._max_plot_images_axfigy)
            in_image_slice_plot = in_image[index].reshape(size_slices)
            in_mask_slice_plot = in_mask[index].reshape(size_slices)

            axis[ind_i, ind_j].set_title('slice %s' % (index))
            axis[ind_i, ind_j].imshow(in_image_slice_plot, cmap='gray')
            axis[ind_i, ind_j].imshow(in_mask_slice_plot, cmap='jet', alpha=0.5)

        if is_save_outfiles:
            outfilename = template_outfilename % ('_'.join([str(x) for x in indexes_slices]))
            plt.savefig(outfilename)
            plt.close()
        else:
            plt.show()

    @classmethod
    def plot_images_masks_all_slices(cls,
                                     in_image: np.ndarray,
                                     in_mask: np.ndarray,
                                     type_output: int = 2,
                                     is_save_outfiles: bool = False,
                                     template_outfilename: str = 'slices_%s.png'
                                     ) -> None:
        num_total_slices = in_image.shape[0]
        size_slices = (in_image.shape[1], in_image.shape[2])

        if type_output == 1:
            num_figures_saved = num_total_slices // (cls._skip_slices_plot_image * cls._max_plot_images_figure)
            range_slices_figure = cls._max_plot_images_figure * cls._skip_slices_plot_image
        elif type_output == 2:
            num_figures_saved = cls._num_figures_saved
            range_slices_figure = num_total_slices // cls._num_figures_saved
        else:
            num_figures_saved = 0
            range_slices_figure = None

        count_slice = 0
        for n in range(num_figures_saved):
            indexes_slices = np.linspace(count_slice, count_slice + range_slices_figure,
                                         num=cls._max_plot_images_figure + 1, dtype=int)[:-1]

            fig, axis = plt.subplots(cls._max_plot_images_axfigx, cls._max_plot_images_axfigy, figsize=(10, 5))

            for i, index in enumerate(indexes_slices):
                (ind_i, ind_j) = (i // cls._max_plot_images_axfigy, i % cls._max_plot_images_axfigy)
                in_image_slice_plot = in_image[index].reshape(size_slices)
                in_mask_slice_plot = in_mask[index].reshape(size_slices)

                axis[ind_i, ind_j].set_title('slice %s' % (index))
                axis[ind_i, ind_j].imshow(in_image_slice_plot, cmap='gray')
                axis[ind_i, ind_j].imshow(in_mask_slice_plot, cmap='jet', alpha=0.5)

            count_slice += range_slices_figure

            if is_save_outfiles:
                outfilename = template_outfilename % ('_'.join([str(x) for x in indexes_slices]))
                plt.savefig(outfilename)
                plt.close()
            else:
                plt.show()

    @classmethod
    def plot_compare_images_masks_all_slices(cls,
                                             in_image: np.ndarray,
                                             original_mask: np.ndarray,
                                             predict_mask: np.ndarray,
                                             type_output: int = 2,
                                             is_save_outfiles: bool = False,
                                             template_outfilename: str = 'slices_%s.png'
                                             ) -> None:
        num_total_slices = in_image.shape[0]
        size_slices = (in_image.shape[1], in_image.shape[2])

        if type_output == 1:
            num_figures_saved = num_total_slices // (cls._skip_slices_plot_image * cls._max_plot_images_axfigy_2)
            range_slices_figure = cls._max_plot_images_axfigy_2 * cls._skip_slices_plot_image
        elif type_output == 2:
            num_figures_saved = cls._num_figures_saved
            range_slices_figure = num_total_slices // cls._num_figures_saved
        else:
            num_figures_saved = 0
            range_slices_figure = None

        count_slice = 0
        for n in range(num_figures_saved):
            indexes_slices = np.linspace(count_slice, count_slice + range_slices_figure,
                                         num=cls._max_plot_images_axfigy_2 + 1, dtype=int)[:-1]

            fig, axis = plt.subplots(2, cls._max_plot_images_axfigy_2, figsize=(10, 5))

            for i, index in enumerate(indexes_slices):
                in_image_slice_plot = in_image[index].reshape(size_slices)
                original_mask_slice_plot = original_mask[index].reshape(size_slices)
                predict_mask_slice_plot = predict_mask[index].reshape(size_slices)

                axis[0, i].set_title('slice %s' % (index))
                axis[0, i].imshow(in_image_slice_plot, cmap='gray')
                axis[0, i].imshow(original_mask_slice_plot, cmap='jet', alpha=0.5)
                axis[1, i].set_title('slice %s' % (index))
                axis[1, i].imshow(in_image_slice_plot, cmap='gray')
                axis[1, i].imshow(predict_mask_slice_plot, cmap='jet', alpha=0.5)

            count_slice += range_slices_figure

            if is_save_outfiles:
                outfilename = template_outfilename % ('_'.join([str(x) for x in indexes_slices]))
                plt.savefig(outfilename)
                plt.close()
            else:
                plt.show()

    @classmethod
    def plot_compare_transform_images_masks_all_slices(cls,
                                                       original_image: np.ndarray,
                                                       original_mask: np.ndarray,
                                                       transform_image: np.ndarray,
                                                       transform_mask: np.ndarray,
                                                       type_output: int = 1,
                                                       is_save_outfiles: bool = False,
                                                       template_outfilename: str = 'slices_%s.png'
                                                       ) -> None:
        num_total_slices = original_image.shape[0]
        size_slices = (original_image.shape[1], original_image.shape[2])

        if type_output == 1:
            num_figures_saved = num_total_slices // (cls._skip_slices_plot_image * cls._max_plot_images_axfigy_2)
            range_slices_figure = cls._max_plot_images_axfigy_2 * cls._skip_slices_plot_image
        elif type_output == 2:
            num_figures_saved = cls._num_figures_saved
            range_slices_figure = num_total_slices // cls._num_figures_saved
        else:
            num_figures_saved = 0
            range_slices_figure = None

        count_slice = 0
        for n in range(num_figures_saved):
            indexes_slices = np.linspace(count_slice, count_slice + range_slices_figure,
                                         num=cls._max_plot_images_axfigy_2 + 1, dtype=int)[:-1]

            fig, axis = plt.subplots(2, cls._max_plot_images_axfigy_2, figsize=(10, 5))

            for i, index in enumerate(indexes_slices):
                original_image_slice_plot = original_image[index].reshape(size_slices)
                original_mask_slice_plot = original_mask[index].reshape(size_slices)
                transform_image_slice_plot = transform_image[index].reshape(size_slices)
                transform_mask_slice_plot = transform_mask[index].reshape(size_slices)

                axis[0, i].set_title('slice %s' % (index))
                axis[0, i].imshow(original_image_slice_plot, cmap='gray')
                axis[0, i].imshow(original_mask_slice_plot, cmap='jet', alpha=0.5)
                axis[1, i].set_title('slice %s' % (index))
                axis[1, i].imshow(transform_image_slice_plot, cmap='gray')
                axis[1, i].imshow(transform_mask_slice_plot, cmap='jet', alpha=0.5)

            count_slice += range_slices_figure

            if is_save_outfiles:
                outfilename = template_outfilename % ('_'.join([str(x) for x in indexes_slices]))
                plt.savefig(outfilename)
                plt.close()
            else:
                plt.show()
