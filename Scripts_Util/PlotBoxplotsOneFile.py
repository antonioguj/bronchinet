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
import seaborn as sns
from collections import OrderedDict
import numpy as np
import argparse



def main(args):

    print("Files to plot (\'%s\')..." %(args.inputfile))

    data_only_string = np.genfromtxt(args.inputfile, dtype=str, delimiter=',')
    fields_names = [item.replace('/', '') for item in data_only_string[0,1:]]
    cases_names  = [item.replace('\'', '') for item in data_only_string[1:,0]]
    num_fields   = len(fields_names)
    num_cases    = len(cases_names)

    data_only_float = np.genfromtxt(args.inputfile, dtype=float, delimiter=',')
    data         = data_only_float[1:,1:]
    data_fields  = [data[:,i] for i in range(len(fields_names))]
    data_cases   = [data[i,:] for i in range(len(cases_names))]


    print("Found fields to plot: \'%s\'..." %(', '.join(map(lambda item: '/'+item+'/', fields_names))))
    #print("Found cases to plot: \'%s\'..." %(', '.join(map(lambda item: '/'+item+'/', cases_names))))

    #plt.boxplot(data, labels=labels)
    sns.boxplot(data=data_fields, palette='Set2', width=0.8)
    sns.swarmplot(data=data_fields, color=".25")
    plt.xticks(plt.xticks()[0], fields_names, size=20)
    plt.yticks(plt.yticks()[0], size=15)
    plt.title(str('image res EXACT Testing'), size=25)
    plt.show()
    #plt.savefig(args.nameoutputfile, format='eps', dpi=1000)
    #plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', type=str)
    parser.add_argument('--nameoutputfile', type=str, default=None)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
