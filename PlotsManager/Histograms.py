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


class Histograms(object):

    @classmethod
    def plot_histogram(cls, image_array,
                       num_bins= 20,
                       isave_outfiles= False,
                       outfilename= 'image_test.png',
                       show_percen_yaxis= False):
        max_val_img = np.max(image_array)
        min_val_img = np.min(image_array)
        bins = np.linspace(min_val_img, max_val_img, num_bins)
        #(hist, bins) = np.histogram(images_array, bins=bins)
        plt.hist(image_array.flatten(), bins=bins, log=True, density=show_percen_yaxis)

        if isave_outfiles:
            plt.savefig(outfilename)
            plt.close()
        else:
            plt.show()


    @classmethod
    def plot_compare_histograms(cls, list_images_array,
                                num_bins=20,
                                isave_outfiles= False,
                                outfilename= 'image_test.png',
                                show_percen_yaxis= False):
        # num_images = len(list_images_array)
        # cmap = plt.get_cmap('rainbow')
        # colors = [cmap(float(i)/(num_images-1)) for i in range(num_images)]
        colors = ['blue', 'red', 'green', 'yellow', 'orange']

        max_val_img =-1.0e+06
        min_val_img = 1.0e+06
        for image_array in list_images_array:
            max_val_img = max(np.max(image_array), max_val_img)
            min_val_img = min(np.min(image_array), min_val_img)
        #endfor
        bins = np.linspace(min_val_img, max_val_img, num_bins)

        for i, image_array in enumerate(list_images_array):
            plt.hist(image_array.flatten(), bins=bins, alpha=0.5, color=colors[i],
                     label='image%s'%(i), log=True, density=show_percen_yaxis)
        #endfor
        plt.legend(loc='upper right')

        if isave_outfiles:
            plt.savefig(outfilename)
            plt.close()
        else:
            plt.show()

