
from common.functionutil import *
#from scipy.stats import ttest_ind as fun_ttest
from scipy.stats import ttest_rel as fun_ttest
import argparse



def main(args):

    cases_names = ['UNet-Lev3', 'UNet-Lev5', 'UNetGNN-Reg', 'UNetGNN-Dyn']

    out_filename = './result_ptest_%s.txt' %(args.infield)

    if args.fromfile:
        if not is_exist_file(args.list_input_files):
            message = "File \'%s\' not found..." %(args.list_input_files)
            catch_error_exception(message)
        fout = open(args.list_input_files, 'r')
        list_input_files = [infile.replace('\n','') for infile in fout.readlines()]
        print("\'input_files\' = %s" % (list_input_files))
    else:
        list_input_files = [infile.replace('\n','') for infile in args.input_files]
    num_data_files = len(list_input_files)


    data_files = []

    for i in range(num_data_files):

        data_file_this       = list_input_files[i]
        raw_data_this_string = np.genfromtxt(data_file_this, dtype=str, delimiter=', ')
        raw_data_this_float  = np.genfromtxt(data_file_this, dtype=float, delimiter=', ')

        fields_names_this = [item.replace('/','') for item in raw_data_this_string[0, 1:]]

        if args.infield not in fields_names_this:
            message = 'field \'%s\' not found in file \'%s\'...' % (args.infield, data_file_this)
            catch_error_exception(message)
        else:
            index_infield_this = fields_names_this.index(args.infield) + 1
            data_files.append(raw_data_this_float[1:, index_infield_this])
    #endfor


    matrix_ptests_files = np.zeros((num_data_files,num_data_files))

    for i in range(num_data_files):
        for j in range(i+1, num_data_files):
            # Compute the p-test crossed between data files i and j (not equal)
            (_,ptest_datai_j) = fun_ttest(data_files[i], data_files[j])
            matrix_ptests_files[i,j] = ptest_datai_j
        #endfor
    #endfor


    # Print out matrix of P-test results
    print("Save in file: \'%s\'..." % (out_filename))
    fout = open(out_filename, 'w')

    strheader = 'cases\t\t|' + '| '.join(['%s\t\t' %(elem) for elem in cases_names]) + '\n'
    fout.write(strheader)
    fout.write('-'*110+'\n')

    for i in range(num_data_files):
        strdata = '%s\t|' %(cases_names[i])
        #fill lower part of output matrix with empty spaces
        for j in range(0, i+1):
            strdata += ' - \t\t\t'
        #endfor
        for j in range(i+1, num_data_files):
            strdata += ' %s\t' %(matrix_ptests_files[i,j])
        #endfor
        strdata += '\n'
        fout.write(strdata)
    #endfor
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=str, nargs='*')
    parser.add_argument('--infield', type=str, default='dice')
    parser.add_argument('--fromfile', type=bool, default=False)
    parser.add_argument('--list_input_files', type=str, default='list_input_files.txt')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)