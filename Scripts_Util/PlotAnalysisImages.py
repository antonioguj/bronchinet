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
from DataLoaders.FileReaders import *
from PlotsManager.General import *
from PlotsManager.Histograms import *
from collections import OrderedDict
import argparse, textwrap


TYPES_PLOT_AVAILABLE = ['histogram']


def main(args):

    save_plot_figures = True

    list_input_files = findFilesDirAndCheck(args.inputdir)

    if not isExistdir(args.outputdir):
        makedir(args.outputdir)

    if args.type == 'histogram':
        template_outfigname = 'fig_hist_%s.png'



    for i, in_file in enumerate(list_input_files):
        print("\nInput: \'%s\'..." % (in_file))

        case_name = basenameNoextension(in_file).split()[0]

        in_image_array = FileReader.get_image_array(in_file)

        if args.type == 'histogram':
            num_bins = 20
            max_val_img = np.max(in_image_array)
            min_val_img = np.min(in_image_array)
            bins = np.linspace(min_val_img, max_val_img, num_bins)

            plt.hist(in_image_array.flatten(), bins=bins, log=True, density=True)
            plt.xlabel('Voxel value', size=10)
            plt.ylabel(case_name, size=10)

        else:
            message = 'type plot \'%s\' not available' % (args.type)
            CatchErrorException(message)

        if save_plot_figures:
            outfigname = template_outfigname % (case_name)
            outfigname = joinpathnames(args.outputdir, outfigname)
            print("Output: \'%s\'..." % (outfigname))

            #plt.savefig(outfigname, format='eps', dpi=1000)
            plt.savefig(outfigname, format='png')
            plt.close()
        else:
            plt.show()
    # endfor



if __name__ == "__main__":
    dict_plots_help = {'histogram': 'plot histograms from images'}
    string_plots_help = '\n'.join([(key + ': ' + val) for key, val in dict_plots_help.items()])

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('inputdir', type=str)
    parser.add_argument('--outputdir', type=str, default='./Out_Figures/')
    parser.add_argument('--type', type=str, default='histogram', help=string_plots_help)
    args = parser.parse_args()

    if args.type not in TYPES_PLOT_AVAILABLE:
        message = 'Type plot chosen \'%s\' not available' % (args.type)
        CatchErrorException(message)

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)