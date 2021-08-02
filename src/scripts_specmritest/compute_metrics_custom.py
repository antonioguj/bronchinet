
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import argparse
import sys

from common.functionutil import basename, join_path_names, list_files_dir, get_substring_filename
from dataloaders.imagefilereader import ImageFileReader
from models.metrics import MeanSquaredError, SNR, PSNR, SSIM
from models.keras.metrics import L1, L2, DSSIM, Perceptual

LIST_METRICS_DEFAULT = ['MSE', 'SNR', 'PSNR', 'SSIM', 'Perceptual']
LIST_METRICS_NEED_TF = ['L1', 'L2', 'DSSIM', 'Perceptual']
PATH_REFERENCE_DEFAULT = './BaseData/Volumes_CUBE_Normalised/'


def main(args):

    list_input_result_files = list_files_dir(args.inputdir)
    # list_input_refer_files = list_files_dir(args.referdir)

    # rearrange "list_type_metrics" with losses that need TF at the end
    list_metrics_need_tf = []
    list_metrics_no_tf = []
    for itype_metric in args.list_type_metrics:
        if itype_metric in LIST_METRICS_NEED_TF:
            list_metrics_need_tf.append(itype_metric)
        else:
            list_metrics_no_tf.append(itype_metric)
    # endfor

    args.list_type_metrics = list_metrics_no_tf + list_metrics_need_tf
    index_metric_convert_tf_tensor = len(list_metrics_no_tf)

    list_metrics = OrderedDict()
    for itype_metric in args.list_type_metrics:
        if itype_metric == 'MSE':
            new_metric = MeanSquaredError()
        elif itype_metric == 'SNR':
            new_metric = SNR()
        elif itype_metric == 'PSNR':
            new_metric = PSNR()
        elif itype_metric == 'SSIM':
            new_metric = SSIM()
        elif itype_metric == 'L1':
            new_metric = L1()
        elif itype_metric == 'L2':
            new_metric = L2()
        elif itype_metric == 'DSSIM':
            new_metric = DSSIM()

        elif itype_metric == 'Perceptual':
            in_image_0_shape = ImageFileReader.get_image_size(list_input_result_files[0])

            if args.is_calc_in_slice:
                if args.orientation_slice == 'axial':
                    in_image_0_shape = (in_image_0_shape[1], in_image_0_shape[2])
                elif args.orientation_slice == 'sagital':
                    in_image_0_shape = (in_image_0_shape[0], in_image_0_shape[1])
                elif args.orientation_slice == 'coronal':
                    in_image_0_shape = (in_image_0_shape[0], in_image_0_shape[2])

            if args.is_calc_rm_border:
                num_slices_rm_border = tuple([int(size * args.prop_rm_border / 2) for size in in_image_0_shape])
                in_image_0_shape = tuple([size - 2 * num for size, num in zip(in_image_0_shape, num_slices_rm_border)])

            new_metric = Perceptual(size_image=in_image_0_shape)
        else:
            print('ERROR: Chosen loss function not found...')
            sys.exit(0)

        list_metrics[itype_metric] = new_metric
    # endfor

    # --------------------

    outdict_calc_metrics = OrderedDict()

    for i, in_result_file in enumerate(list_input_result_files):
        print("\nInput: \'%s\'..." % (basename(in_result_file)))
        in_refer_file = basename(in_result_file).replace('GRASE', 'CUBE').replace('_probmap', '')
        in_refer_file = join_path_names(args.referdir, in_refer_file)
        print("And: \'%s\'..." % (basename(in_refer_file)))

        in_result_image = ImageFileReader.get_image(in_result_file)
        in_refer_image = ImageFileReader.get_image(in_refer_file)

        # --------------------

        if args.is_calc_in_slice:
            print('To compute metrics, using the 2D slice \'%s\' extracted in \'%s\' dimension...'
                  % (args.index_slice, args.orientation_slice))

            if args.orientation_slice == 'axial':
                num_slices = in_result_image.shape[0]
            elif args.orientation_slice == 'sagital':
                num_slices = in_result_image.shape[2]
            elif args.orientation_slice == 'coronal':
                num_slices = in_result_image.shape[1]
            else:
                print('ERROR: Wrong input \'orientation\'. Only available: [\'axial\', \'sagital\', \'coronal\']...')
                sys.exit(0)

            if args.index_slice == 'middle':
                args.index_slice = int(num_slices / 2)

            print('2D slice taken from slice \'%s\'...' % (args.index_slice))

            if args.orientation_slice == 'axial':
                in_result_image = in_result_image[args.index_slice, :, :]
                in_refer_image = in_refer_image[args.index_slice, :, :]
            elif args.orientation_slice == 'sagital':
                in_result_image = in_result_image[:, :, args.index_slice]
                in_refer_image = in_refer_image[:, :, args.index_slice]
            elif args.orientation_slice == 'coronal':
                in_result_image = in_result_image[:, args.index_slice, :]
                in_refer_image = in_refer_image[:, args.index_slice, :]

        # --------------------

        if args.is_calc_rm_border:
            print('To compute metrics, remove borders of volume, with proportion of \'%s\'...' % (args.prop_rm_border))

            num_slices_rm_border = tuple([int(size * args.prop_rm_border / 2) for size in in_result_image.shape])
            print('Remove num slices in each side in dirs \'%s\'...' % (str(num_slices_rm_border)))

            if len(in_result_image.shape) == 2:
                in_result_image = in_result_image[num_slices_rm_border[0]: -num_slices_rm_border[0],
                                                  num_slices_rm_border[1]: -num_slices_rm_border[1]]
                in_refer_image = in_refer_image[num_slices_rm_border[0]: -num_slices_rm_border[0],
                                                num_slices_rm_border[1]: -num_slices_rm_border[1]]

            elif len(in_result_image.shape) == 3:
                in_result_image = in_result_image[num_slices_rm_border[0]: -num_slices_rm_border[0],
                                                  num_slices_rm_border[1]: -num_slices_rm_border[1],
                                                  num_slices_rm_border[2]: -num_slices_rm_border[2]]
                in_refer_image = in_refer_image[num_slices_rm_border[0]: -num_slices_rm_border[0],
                                                num_slices_rm_border[1]: -num_slices_rm_border[1],
                                                num_slices_rm_border[2]: -num_slices_rm_border[2]]

            print('Result images of size: \'%s\'...' % (str(in_result_image.shape)))

        # --------------------

        print("\nCompute the Metrics:")
        casename = get_substring_filename(basename(in_refer_file), pattern_search='Sujeto[0-9]+-[a-z]+')
        outdict_calc_metrics[casename] = []

        for ind, (itype_metric, imetric) in enumerate(list_metrics.items()):

            if ind == index_metric_convert_tf_tensor:
                # all metrics thereafter need tf tensors (rearranged "list_metrics" above)
                in_result_image = np.reshape(in_result_image, (1,) + in_result_image.shape + (1,))
                in_refer_image = np.reshape(in_refer_image, (1,) + in_refer_image.shape + (1,))

                in_result_image = tf.convert_to_tensor(in_result_image, tf.float32)
                in_refer_image = tf.convert_to_tensor(in_refer_image, tf.float32)

            if itype_metric in LIST_METRICS_NEED_TF:
                out_metric = imetric.compute(in_refer_image, in_result_image)
                outval_metric = out_metric.numpy()
            else:
                outval_metric = imetric.compute(in_refer_image, in_result_image)

            print("\'%s\': %s..." % (itype_metric, outval_metric))
            outdict_calc_metrics[casename].append(outval_metric)
        # endfor
    # endfor

    # --------------------

    # write out computed metrics in file
    with open(args.output_file, 'w') as fout:
        strheader = ', '.join(['/case/'] + ['/%s/' % (key) for key in list_metrics.keys()]) + '\n'
        fout.write(strheader)

        for (in_casename, outlist_calc_metrics) in outdict_calc_metrics.items():
            list_write_data = [in_casename] + ['%0.6f' % (elem) for elem in outlist_calc_metrics]
            strdata = ', '.join(list_write_data) + '\n'
            fout.write(strdata)
        # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('--referdir', type=str, default=PATH_REFERENCE_DEFAULT)
    parser.add_argument('--list_type_metrics', type=str, nargs='*', default=LIST_METRICS_DEFAULT)
    parser.add_argument('--output_file', type=str, default='./result_metrics.csv')
    parser.add_argument('--is_calc_in_slice', type=bool, default=False)
    parser.add_argument('--index_slice', type=str, default='middle')
    parser.add_argument('--orientation_slice', type=str, default='axial')
    parser.add_argument('--is_calc_rm_border', type=bool, default=False)
    parser.add_argument('--prop_rm_border', type=float, default=0.25)
    args = parser.parse_args()

    if 'Perceptual' in args.list_type_metrics and not args.is_calc_rm_border:
        print('WARNING: \'Perceptual\' is in \'list_type_metrics\'. Enable \'is_calc_rm_border\' with proportion 0.1..')
        args.is_calc_rm_border = True
        args.prop_rm_border = 0.1

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
