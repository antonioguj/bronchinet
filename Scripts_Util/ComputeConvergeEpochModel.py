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
from collections import *
import numpy as np
import argparse



def main(args):
    # ---------- SETTINGS ----------
    type_eval_loss = 'mean'
    type_converged_epoch = 'last'
    tolerance_converge = 0.001
    tolerance_diverge = 0.05
    num_epochs_average = 20
    num_epochs_patience = 1
    num_epochs_jumpeval = 1
    writeout_meanlosshistory = False
    filename_meanlosshistory = args.inlossHistory.replace('.txt','') + '_%s_aver%s_jump%s.txt' %(type_eval_loss,
                                                                                                 num_epochs_average,
                                                                                                 num_epochs_jumpeval)
    # ---------- SETTINGS ----------


    with open(args.inlossHistory, 'r') as infile:
        header_line = infile.readline()
        header_items = map(lambda item: item.replace('/', '').replace('\n', ''), header_line.split(' '))

    data_file = np.loadtxt(args.inlossHistory, skiprows=1)
    epochs = data_file[:,0]
    train_loss = data_file[:,1]
    valid_loss = data_file[:,2]


    first_epoch_eval = num_epochs_average
    first_epoch_compare = first_epoch_eval + num_epochs_patience
    if len(epochs) < first_epoch_compare:
        print "ERROR: loss history not long enough: %s < %s... EXIT" %(len(epochs), first_epoch_compare)
        exit(0)
    else:
        last_epoch_compare = len(epochs)
        last_epoch_eval = last_epoch_compare - num_epochs_patience


    num_evals_loss = int((last_epoch_compare - first_epoch_eval) / num_epochs_jumpeval) #+ 1
    if num_epochs_jumpeval>1:
        num_evals_loss += 1
        
    evalarr_epochs = np.zeros(num_evals_loss, dtype=int)
    evalarr_train_loss = np.zeros(num_evals_loss, dtype=float)
    evalarr_valid_loss = np.zeros(num_evals_loss, dtype=float)

    for count, ind_epoch in enumerate(range(first_epoch_eval, last_epoch_compare, num_epochs_jumpeval)):
        beg_epoch_average = ind_epoch - num_epochs_average
        end_epoch_average = ind_epoch

        evalarr_epochs[count] = ind_epoch
        if type_eval_loss == 'mean':
            evalarr_train_loss[count] = np.mean(train_loss[beg_epoch_average:end_epoch_average])
            evalarr_valid_loss[count] = np.mean(valid_loss[beg_epoch_average:end_epoch_average])
        elif type_eval_loss == 'min':
            evalarr_train_loss[count] = np.min(train_loss[beg_epoch_average:end_epoch_average])
            evalarr_valid_loss[count] = np.min(valid_loss[beg_epoch_average:end_epoch_average])
        elif type_eval_loss == 'std':
            evalarr_train_loss[count] = np.std(train_loss[beg_epoch_average:end_epoch_average])
            evalarr_valid_loss[count] = np.std(valid_loss[beg_epoch_average:end_epoch_average])
    #endfor


    if writeout_meanlosshistory:
        print("Write out computed mean loss History in file: %s..." %(filename_meanlosshistory))
        fout = open(filename_meanlosshistory, 'w')
        fout.write('/epoch/ /loss/ /val_loss/\n')
        for i in range(num_evals_loss):
            strline = '%s %s %s\n' %(evalarr_epochs[i], evalarr_train_loss[i], evalarr_valid_loss[i])
            fout.write(strline)
        #endfor
        fout.close()


    jump_elems_patiente = num_epochs_patience / num_epochs_jumpeval

    is_converge_found = False
    epoch_converged = None
    epoch_diverged = None
    for count, ind_epoch in enumerate(range(first_epoch_eval, last_epoch_eval, num_epochs_jumpeval)):
        count_compare = count + jump_elems_patiente

        eval_train_loss = evalarr_train_loss[count]
        eval_valid_loss = evalarr_valid_loss[count]
        compare_train_loss = evalarr_train_loss[count_compare]
        compare_valid_loss = evalarr_valid_loss[count_compare]

        #absdiff_train_loss = (eval_train_loss - compare_train_loss)
        #absdiff_valid_loss = (eval_valid_loss - compare_valid_loss)
        #reldiff_train_loss = (eval_train_loss - compare_train_loss) / (eval_train_loss + 1.0e-12)
        reldiff_valid_loss = (eval_valid_loss - compare_valid_loss) / (eval_valid_loss + 1.0e-12)

        print compare_valid_loss
        print eval_valid_loss

        #print ind_epoch, reldiff_valid_loss
        #print
        if reldiff_valid_loss > 0.0:
            if reldiff_valid_loss < tolerance_converge:
                is_converge_found = True
                ind_epoch_lastavail = ind_epoch + jump_elems_patiente
                epoch_converged = ind_epoch_lastavail
                epoch_minvalidloss = 1 + np.argmin(valid_loss[:ind_epoch_lastavail])
                validloss_converged = valid_loss[epoch_converged-1]
                validloss_min = valid_loss[epoch_minvalidloss-1]
                print "CONVERGED at EPOCH \'%s\' WITH VALIDLOSS \'%s\' AND RELDIFF_LOSS \'%s\'. MIN VALIDLOSS at EPOCH \'%s\': \'%s\'..."\
                      %(epoch_converged, validloss_converged, reldiff_valid_loss, epoch_minvalidloss, validloss_min)
        else:
            if abs(reldiff_valid_loss) > tolerance_diverge:
                is_converge_found = False
                ind_epoch_lastavail = ind_epoch + jump_elems_patiente
                epoch_diverged = ind_epoch_lastavail
                epoch_minvalidloss = 1 + np.argmin(valid_loss[:ind_epoch_lastavail])
                validloss_diverged = valid_loss[epoch_diverged-1]
                validloss_min = valid_loss[epoch_minvalidloss-1]
                print "DIVERGED at EPOCH \'%s\' WITH VALIDLOSS \'%s\' AND RELDIFF_LOSS \'%s\'. MIN VALIDLOSS at EPOCH \'%s\': \'%s\'..." \
                      %(epoch_diverged, validloss_diverged, reldiff_valid_loss, epoch_minvalidloss, validloss_min)
                #break
    #endfor




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inlossHistory', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
