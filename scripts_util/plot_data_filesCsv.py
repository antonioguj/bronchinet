
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from common.functionutil import *
import argparse

TYPES_PLOT_AVAILABLE = ['plot', 'scatter', 'boxplot']



def main(args):

    if args.fromfile:
        if not is_exist_file(args.list_input_files):
            message = "File \'%s\' not found..." %(args.list_input_files)
            catch_error_exception(message)
        fout = open(args.list_input_files, 'r')
        list_input_files = [infile.replace('\n','') for infile in fout.readlines()]
        print("\'input_files\' = %s" % (list_input_files))
    else:
        list_input_files = [infile.replace('\n','') for infile in args.input_files]
    num_input_files = len(list_input_files)

    print("Files to plot data from: \'%s\'..." %(num_input_files))
    for i, ifile in enumerate(list_input_files):
        print("%s: \'%s\'" %(i+1, ifile))
    # endfor


    # ---------- SETTINGS ----------
    labels = ['model_%i'%(i+1) for i in range(num_input_files)]
    #labels = ['Refer_KNN+RG+VB',
    #          'Unet_LUVAR',
    #          'Unet_DLCST+LUVAR']
    titles = ['Distance False Positives', 'Distance False Negatives']
    save_plot_figures = False
    template_outfigname = 'fig_%s' %(args.type) + '_%s.png'
    # ---------- SETTINGS ----------



    dict_data_fields_files = OrderedDict()

    if args.alldata_oneplot:
        print("Plot all data from input file in one plot...")
        in_file = list_input_files[0]

        raw_data_this_string = np.genfromtxt(in_file, dtype=str, delimiter=', ')
        raw_data_this_float  = np.genfromtxt(in_file, dtype=float, delimiter=', ')

        header_this = list(raw_data_this_string[0, :])
        list_fields = [elem.replace('/', '') for elem in header_this[1:]]
        data_this   = raw_data_this_float[1:, 1:]

        labels = list_fields

        dict_data_fields_files['All'] = []
        # store data from this file
        for i, ifield in enumerate(list_fields):
            dict_data_fields_files['All'].append(data_this[:, i])
        # endfor

    else:
        for i, in_file in enumerate(list_input_files):

            raw_data_this_string = np.genfromtxt(in_file, dtype=str, delimiter=', ')
            raw_data_this_float  = np.genfromtxt(in_file, dtype=float, delimiter=', ')

            header_this    = list(raw_data_this_string[0, :])
            rows1elem_this = list(raw_data_this_string[:, 0])
            data_this      = raw_data_this_float[1:, 1:]

            if i == 0:
                header_file     = header_this
                rows1elem_file  = rows1elem_this
                list_fields_file= [elem.replace('/', '') for elem in header_file[1:]]

                for ifield in list_fields_file:
                    dict_data_fields_files[ifield] = []
                # endfor
            else:
                if header_this != header_file:
                    message = 'header in file: \'%s\' not equal to header found previously: \'%s\'' % (header_this, header_file)
                    catch_error_exception(message)
                # if rows1elem_this != rows1elem_file:
                #    message = '1st column in file: \'%s\' not equal to 1st column found previously: \'%s\'' % (rows1elem_this, rows1elem_file)
                #    CatchErrorException(message)

            # store data from this file
            for i, ifield in enumerate(list_fields_file):
                dict_data_fields_files[ifield].append(data_this[:, i])
            # endfor
        # endfor

    print("Found fields to plot data from: %s..." % (dict_data_fields_files.keys()))



    for (ifield, data_files) in dict_data_fields_files.items():

        if args.type == 'plot':
            for i, idata in enumerate(data_files):
                xrange = range(1, len(idata)+1)
                plt.plot(xrange, idata, label=labels[i])
            # endfor
            plt.xlabel('Num cases', size=10)
            plt.ylabel(ifield.title(), size=15)
            plt.legend(loc='best')

        elif args.type == 'scatter':
            for i, idata in enumerate(data_files):
                xrange = range(1, len(idata)+1)
                plt.scatter(xrange, idata, label=labels[i])
            # endfor
            plt.xlabel('Num cases', size=10)
            plt.ylabel(ifield.title(), size=15)
            plt.legend(loc='best')

        elif args.type == 'boxplot':
            # plt.boxplot(data_files, labels=labels)
            sns.boxplot  (data=data_files, palette='Set2', width=0.8)
            sns.swarmplot(data=data_files, color=".25")
            plt.xticks(plt.xticks()[0], labels, size=10)
            plt.yticks(plt.yticks()[0], size=15)
            plt.ylabel(ifield.title(), size=15)

        else:
            message = 'type plot \'%s\' not available' % (args.type)
            catch_error_exception(message)

        if save_plot_figures:
            outfigname = template_outfigname % (ifield)
            print("Output: \'%s\'..." % (outfigname))

            #plt.savefig(outfigname, format='eps', dpi=1000)
            plt.savefig(outfigname, format='png')
            plt.close()
        else:
            plt.show()
    # endfor



if __name__ == "__main__":
    dict_plots_help = {'plot': 'plot data',
                       'scatter': 'plot scattered data points',
                       'boxplot': 'plot boxplots from data'}
    string_plots_help = '\n'.join([(key + ': ' + val) for key, val in dict_plots_help.items()])

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_files', type=str, nargs='*')
    parser.add_argument('--type', type=str, default='boxplot', help=string_plots_help)
    parser.add_argument('--fromfile', type=bool, default=False)
    parser.add_argument('--list_input_files', type=str, default='list_input_files.txt')
    parser.add_argument('--alldata_oneplot', type=str, default=False)
    args = parser.parse_args()

    if args.type not in TYPES_PLOT_AVAILABLE:
        message = 'Type plot chosen \'%s\' not available' % (args.type)
        catch_error_exception(message)

    if args.fromfile and not args.list_input_files:
        message = 'need to input \'list_input_files\' with filenames to plot'
        catch_error_exception(message)

    if args.alldata_oneplot and len(args.input_files) > 1:
        message = 'when plotting all data in one plot, only need to input one file'
        catch_error_exception(message)

    if args.alldata_oneplot and args.fromfile:
        message = 'when plotting all data in one plot, cannot read input files from file'
        catch_error_exception(message)

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)