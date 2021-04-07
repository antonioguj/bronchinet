
from typing import List
import matplotlib.pyplot as plt
import numpy as np


class Histogram(object):

    @classmethod
    def get_histogram_data(cls,
                           in_image: np.ndarray,
                           num_bins: int = 20,
                           density_range: bool = False
                           ) -> np.ndarray:
        max_value = np.max(in_image)
        min_value = np.min(in_image)
        bins = np.linspace(min_value, max_value, num_bins)
        (out_hist_data, bins) = np.histogram(in_image, bins=bins, density=density_range)
        return out_hist_data

    @classmethod
    def plot_histogram(cls,
                       in_image: np.ndarray,
                       num_bins: int = 20,
                       density_range: bool = False,
                       is_save_outfiles: bool = False,
                       outfilename: str = 'image_test.png'
                       ) -> None:
        max_value = np.max(in_image)
        min_value = np.min(in_image)
        bins = np.linspace(min_value, max_value, num_bins)
        plt.hist(in_image.flatten(), bins=bins, log=True, density=density_range)

        if is_save_outfiles:
            plt.savefig(outfilename)
            plt.close()
        else:
            plt.show()

    @classmethod
    def plot_compare_histograms(cls,
                                in_list_images: List[np.ndarray],
                                num_bins: int = 20,
                                density_range: bool = False,
                                is_save_outfiles: bool = False,
                                outfilename: str = 'image_test.png'
                                ) -> None:
        # num_images = len(list_images)
        # cmap = plt.get_cmap('rainbow')
        # colors = [cmap(float(i)/(num_images-1)) for i in range(num_images)]
        colors = ['blue', 'red', 'green', 'yellow', 'orange']

        max_value = -1.0e+06
        min_value = 1.0e+06
        for in_image in in_list_images:
            max_value = max(np.max(in_image), max_value)
            min_value = min(np.min(in_image), min_value)

        bins = np.linspace(min_value, max_value, num_bins)

        for i, in_image in enumerate(in_list_images):
            plt.hist(in_image.flatten(), bins=bins, alpha=0.5, color=colors[i],
                     label='image%s' % (i), log=True, density=density_range)

        plt.legend(loc='upper right')
        if is_save_outfiles:
            plt.savefig(outfilename)
            plt.close()
        else:
            plt.show()
