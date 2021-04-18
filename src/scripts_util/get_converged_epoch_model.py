
import numpy as np
import argparse

from common.functionutil import calc_moving_average
from common.exceptionmanager import catch_error_exception


def main(args):

    # SETTINGS
    if args.is_output_aver_loss and args.is_move_aver:
        out_aver_losshist_filename = args.input_loss_file.replace('.csv', '') + '_aver-%d.csv' % (args.size_aver)
    else:
        out_aver_losshist_filename = None
    # --------

    epochs = None
    data_fields = None
    index_field_eval = None
    data_eval_original = None

    with open(args.input_loss_file, 'r') as infile:
        header_line = infile.readline()
        header_items = [elem.replace('/', '').replace('\n', '') for elem in header_line.split(' ')]

        if args.field_eval not in header_items:
            message = 'field chosen to evaluate the convergence of losses (\'%s\') not found in loss file' \
                      % (args.field_eval)
            catch_error_exception(message)
        else:
            index_field_eval = header_items.index(args.field_eval) - 1

        data_loss_file = np.loadtxt(args.input_loss_file, skiprows=1, delimiter=' ')
        epochs = data_loss_file[:, 0].astype(np.uint16)
        data_fields = data_loss_file[:, 1:]
        data_eval_original = data_fields[:, index_field_eval]

    if args.is_move_aver:
        print('Compute the moving average of the losses, with size of average window \'%s\'...' % (args.size_aver))
        # for 'epochs': decrease the num epochs by the size of average window
        num_epochs = len(epochs)
        num_aver_epochs = num_epochs - args.size_aver + 1
        epochs = epochs[0:num_aver_epochs]

        # for other fields: compute the moving average of the list of values
        num_fields = data_fields.shape[1]
        aver_data_fields = np.ndarray((num_aver_epochs, num_fields), dtype=np.float32)
        for i in range(num_fields):
            aver_data_fields[:, i] = calc_moving_average(data_fields[:, i], args.size_aver)
        # endfor

        data_fields = aver_data_fields

    if args.is_output_aver_loss and args.is_move_aver:
        print("Write out the computed moving average of the losses in file: %s...\n" % (out_aver_losshist_filename))
        fout = open(out_aver_losshist_filename, 'w')

        with open(args.input_loss_file, 'r') as infile:
            header_line = infile.readline()
            fout.write(header_line)

        num_aver_epochs = len(epochs)
        for i in range(num_aver_epochs):
            list_strdata = ['%d' % (epochs[i])] + ['%0.6f' % (elem) for elem in list(data_fields[i, :])]
            writeline = ' '.join(list_strdata) + '\n'
            fout.write(writeline)
        # endfor
        fout.close()

    # ******************************

    data_eval = data_fields[:, index_field_eval]

    print('Evaluate the relative difference of the \'%s\' history, between epochs with patience \'%s\'...'
          % (args.field_eval, args.patience_converge))
    print('Thresholds to mark convergence: %s, and divergence: %s...' % (args.thres_converge, args.thres_diverge))
    is_found_converge = False
    is_found_diverge = False
    epoch_converged = None
    epoch_diverged = None

    list_epochs_evaluate = list(epochs[args.patience_converge:])

    for i_epoch_eval in list_epochs_evaluate:
        i_epoch_original = i_epoch_eval + args.size_aver - 1

        value_eval_this = data_eval[i_epoch_eval - 1]
        value_eval_compare = data_eval[i_epoch_eval - args.patience_converge - 1]
        # value_original = data_eval_original[i_epoch_original - 1]

        relerror_diff_values = abs((value_eval_this - value_eval_compare) / (value_eval_this + 1.0e-12))

        # print("epoch \'%s\' (original \'%s\'): averaged value \'%s\' (original \'%s\'), relative error \'%0.6f\' ..."
        #       % (i_epoch_eval, i_epoch_original, value_eval_this, value_original, relerror_diff_values))

        if relerror_diff_values > 0.0:
            if relerror_diff_values < args.thres_converge:
                is_found_converge = True
                epoch_converged = i_epoch_original
                epoch_min_value = 1 + np.argmin(data_eval_original[:epoch_converged])
                value_converged = data_eval_original[epoch_converged - 1]
                value_min_found = data_eval_original[epoch_min_value - 1]

                print("CONVERGED: at epoch \'%s\' with value \'%s\' and relative error \'%0.6f\'"
                      ". Min value found until here: \'%s\', at epoch \'%s\'...\n"
                      % (epoch_converged, value_converged, relerror_diff_values, value_min_found, epoch_min_value))
                # break
        else:
            if relerror_diff_values > args.thres_diverge:
                is_found_diverge = False
                epoch_diverged = i_epoch_original
                value_diverged = data_eval_original[epoch_diverged - 1]
                print("DIVERGED: at epoch \'%s\' with value \'%s\' and relative error \'%0.6f\'...\n"
                      % (epoch_diverged, value_diverged, relerror_diff_values))
                # break
    # endfor

    if is_found_converge:
        print("\nGOOD: HISTORY OF \'%s\' IS CONVERGED AT EPOCH \'%s\'..." % (args.field_eval, epoch_converged))
    elif is_found_diverge:
        print("\nBAD: HISTORY OF \'%s\' IS DIVERGED AT EPOCH \'%s\'..." % (args.field_eval, epoch_diverged))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_loss_file', type=str)
    parser.add_argument('--field_eval', type=str, default='val_loss')
    parser.add_argument('--patience_converge', type=int, default=20)
    parser.add_argument('--thres_converge', type=float, default=0.001)
    parser.add_argument('--thres_diverge', type=float, default=0.05)
    parser.add_argument('--is_move_aver', type=bool, default=True)
    parser.add_argument('--size_aver', type=int, default=50)
    parser.add_argument('--is_output_aver_loss', type=bool, default=False)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
