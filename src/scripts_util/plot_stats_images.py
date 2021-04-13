
import numpy as np
import matplotlib.pyplot as plt
import argparse

from common.functionutil import makedir, is_exist_dir, join_path_names, basename_filenoext, list_files_dir
from common.exceptionmanager import catch_error_exception
from dataloaders.imagefilereader import ImageFileReader
# from plotting.histogram import Histogram

TYPES_PLOT_AVAILABLE = ['histogram']


def main(args):

    # SETTINGS
    save_plot_figures = False
    # --------

    list_input_files = list_files_dir(args.input_dir)

    if not is_exist_dir(args.output_dir):
        makedir(args.output_dir)

    if args.type_plot == 'histogram':
        template_outfigname = 'fig_hist_%s.png'
    else:
        template_outfigname = None

    # ******************************

    for in_file in list_input_files:
        print("\nInput: \'%s\'..." % (in_file))

        case_name = basename_filenoext(in_file).split()[0]

        in_image = ImageFileReader.get_image(in_file)

        if args.type_plot == 'histogram':
            num_bins = 20
            max_val_img = np.max(in_image)
            min_val_img = np.min(in_image)
            bins = np.linspace(min_val_img, max_val_img, num_bins)

            plt.hist(in_image.flatten(), bins=bins, log=False, density=True)
            plt.xlabel('Voxel value', size=10)
            plt.ylabel(case_name, size=10)

        else:
            message = 'type plot \'%s\' not available' % (args.type_plot)
            catch_error_exception(message)

        if save_plot_figures:
            outfigname = template_outfigname % (case_name)
            outfigname = join_path_names(args.output_dir, outfigname)
            print("Output: \'%s\'..." % (outfigname))

            # plt.savefig(outfigname, format='eps', dpi=1000)
            plt.savefig(outfigname, format='png')
            plt.close()
        else:
            plt.show()
    # endfor


if __name__ == "__main__":
    dict_plots_help = {'histogram': 'plot histograms from images'}
    string_plots_help = '\n'.join([(key + ': ' + val) for key, val in dict_plots_help.items()])

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='./Out_Figures/')
    parser.add_argument('--type_plot', type=str, default='histogram', help=string_plots_help)
    args = parser.parse_args()

    if args.type_plot not in TYPES_PLOT_AVAILABLE:
        message = 'Type plot chosen \'%s\' not available' % (args.type_plot)
        catch_error_exception(message)

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
