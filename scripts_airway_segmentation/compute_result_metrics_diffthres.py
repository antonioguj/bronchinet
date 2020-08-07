
from common.constant import *
from common.functionutil import *
from common.workdirmanager import TrainDirManager
from dataloaders.imagefilereader import ImageFileReader
from models.metrics import get_metric
from imageoperators.imageoperator import MorphoDilateMask, ThresholdImage, ThinningMask, FirstConnectedRegionMask
from imageoperators.maskoperator import MaskOperator
from collections import OrderedDict
from tqdm import tqdm
import argparse
np.random.seed(2017)



def main(args):
    # ---------- SETTINGS ----------
    outfilename_metrics_eachcase = 'results_metrics_diffThres_%s.csv'
    outfilename_metrics_meanall  = 'meanAllcases_results_metrics_diffThres.csv'

    # parameters to draw ROC curve
    inlist_thresholds = [0.0]
    # num_thresholds = 9
    range_threshold = [-6, -2]
    inlist_thresholds += (np.logspace(range_threshold[0], range_threshold[1], 5)).tolist()
    # num_thresholds = 9
    range_threshold = [0.1, 0.5]
    inlist_thresholds += (np.linspace(range_threshold[0], range_threshold[1], 5)).tolist()
    # num_thresholds = 9
    # range_threshold = [-2, -10]
    # inlist_thresholds += [1.0 - elem for elem in (np.logspace(range_threshold[0], range_threshold[1], num_thresholds)).tolist()]
    # #allowedDistance = 0
    #inlist_thresholds += [1.0]
    print("List of Threshold values: %s" % (inlist_thresholds))
    # ---------- SETTINGS ----------


    workdir_manager                 = TrainDirManager(args.basedir)
    input_posteriors_path           = workdir_manager.get_pathdir_exist(args.input_posteriors_dir)
    input_reference_masks_path      = workdir_manager.get_datafile_exist(args.name_input_reference_masks_relpath)
    output_files_path               = workdir_manager.get_pathdir_new(args.output_dir)
    list_input_posteriors_files     = list_files_dir(input_posteriors_path)
    list_input_reference_masks_files= list_files_dir(input_reference_masks_path)
    prefix_pattern_input_files      = get_prefix_pattern_filename(list_input_reference_masks_files[0])

    if (args.is_remove_trachea_calc_metrics):
        input_coarse_airways_path       = workdir_manager.get_datafile_exist(args.name_input_coarse_airways_relpath)
        list_input_coarse_airways_files = list_files_dir(input_coarse_airways_path)


    list_metrics_compute = OrderedDict()
    list_is_use_reference_cenlines = []
    list_is_use_predicted_cenlines = []
    for itype_metric in args.list_type_metrics_ROC_curve:
        new_metric = get_metric(itype_metric)
        list_metrics_compute[new_metric._name_fun_out] = new_metric.compute

        list_is_use_reference_cenlines.append(new_metric._is_use_ytrue_cenlines)
        list_is_use_predicted_cenlines.append(new_metric._is_use_ypred_cenlines)
    # endfor

    is_load_reference_cenlines_files = any(list_is_use_reference_cenlines)
    is_load_predicted_cenlines_files = any(list_is_use_predicted_cenlines)

    if (is_load_reference_cenlines_files):
        print("Loading Reference Centrelines...")
        input_reference_cenlines_path       = workdir_manager.get_datafile_exist(args.name_input_reference_centrelines_relpath)
        list_input_reference_cenlines_files = list_files_dir(input_reference_cenlines_path)


    num_input_files  = len(list_input_posteriors_files)
    num_comp_metrics = len(list_metrics_compute)
    num_thresholds   = len(inlist_thresholds)
    out_computed_metrics_allcases = np.zeros((num_input_files, num_thresholds, num_comp_metrics))


    for i, in_posterior_file in enumerate(list_input_posteriors_files):
        print("\nInput: \'%s\'..." % (basename(in_posterior_file)))

        in_reference_mask_file = find_file_inlist_same_prefix(basename(in_posterior_file), list_input_reference_masks_files,
                                                              prefix_pattern=prefix_pattern_input_files)
        print("Reference mask file: \'%s\'..." % (basename(in_reference_mask_file)))

        in_posterior = ImageFileReader.get_image(in_posterior_file)
        in_reference_mask = ImageFileReader.get_image(in_reference_mask_file)
        print("Predictions of size: %s..." % (str(in_posterior.shape)))

        if (is_load_reference_cenlines_files):
            in_reference_cenline_file = find_file_inlist_same_prefix(basename(in_posterior_file), list_input_reference_cenlines_files,
                                                                     prefix_pattern=prefix_pattern_input_files)
            print("Reference centrelines file: \'%s\'..." % (basename(in_reference_cenline_file)))
            in_reference_cenline = ImageFileReader.get_image(in_reference_cenline_file)


        if (args.is_remove_trachea_calc_metrics):
            print("Remove trachea and main bronchii masks in computed metrics...")
            in_coarse_airways_file = find_file_inlist_same_prefix(basename(in_posterior_file), list_input_coarse_airways_files,
                                                                  prefix_pattern=prefix_pattern_input_files)
            print("Coarse Airways mask file: \'%s\'..." % (basename(in_coarse_airways_file)))

            in_coarse_airways = ImageFileReader.get_image(in_coarse_airways_file)

            print("Dilate coarse airways masks 4 levels to remove completely trachea and main bronchi from ground-truth...")
            in_coarse_airways = MorphoDilateMask.compute(in_coarse_airways, num_iters=4)

            in_reference_mask = MaskOperator.substract_two_masks(in_reference_mask, in_coarse_airways)

            if (is_load_reference_cenlines_files):
                in_reference_cenline = MaskOperator.substract_two_masks(in_reference_cenline, in_coarse_airways)


        # *******************************************************************************
        # Compute and store Metrics at all thresholds
        print("\nCompute Metrics with all thresholds \'%s\'..." %(inlist_thresholds))
        print("- Compute the binary masks by thresholding the posteriors...")
        if args.is_connected_masks:
            print("- Compute the first connected component from the binary masks...")
        if args.is_remove_trachea_calc_metrics:
            print("- Remove the trachea and main bronchii from the binary masks...")
        if is_load_predicted_cenlines_files:
            print("- Compute the centrelines by thinning the binary masks...")
        print("- Compute the Metrics:")

        key_casename = get_substring_filename(basename(in_posterior_file), substr_pattern=prefix_pattern_input_files)[:-1]
        outdict_computed_metrics = OrderedDict()

        with tqdm(total=num_thresholds) as progressbar:
            for j, in_thres_value in enumerate(inlist_thresholds):

                # Compute the binary masks by thresholding the posteriors
                in_predicted_mask = ThresholdImage.compute(in_posterior, in_thres_value)

                if (args.is_connected_masks):
                    # First, attach the trachea and main bronchii to the binary masks, to be able to compute the largest connected component
                    in_predicted_mask = MaskOperator.merge_two_masks(in_predicted_mask, in_coarse_airways)  # isNot_intersect_masks=True)

                    # Compute the first connected component from the binary masks
                    in_predicted_mask = FirstConnectedRegionMask.compute(in_predicted_mask, connectivity_dim=1)

                if (is_load_predicted_cenlines_files):
                    # Compute centrelines by thinning the binary masks
                    try:
                        #'try' statement to 'catch' issues when predictions are 'weird' (for extreme threshold values)
                        in_predicted_cenline = ThinningMask.compute(in_predicted_mask)
                    except:
                        in_predicted_cenline = np.zeros_like(in_predicted_mask)

                if (args.is_remove_trachea_calc_metrics):
                    # Remove the trachea and main bronchii from the binary masks
                    in_predicted_mask = MaskOperator.substract_two_masks(in_predicted_mask, in_coarse_airways)

                    if (is_load_predicted_cenlines_files):
                        in_predicted_cenline = MaskOperator.substract_two_masks(in_predicted_cenline, in_coarse_airways)
                # **********************************************


                outdict_computed_metrics[in_thres_value] = []
                for k, (imetric_name, imetric_compute) in enumerate(list_metrics_compute.items()):
                    if list_is_use_reference_cenlines[k]:
                        in_reference_data = in_reference_cenline
                    else:
                        in_reference_data = in_reference_mask
                    if list_is_use_predicted_cenlines[k]:
                        in_predicted_data = in_predicted_cenline
                    else:
                        in_predicted_data = in_predicted_mask

                    try:
                        #'try' statement to 'catch' issues when predictions are 'null' (for extreme threshold values)
                        out_val_metric = imetric_compute(in_reference_data, in_predicted_data)
                    except:
                        # set dummy value for cases with issues
                        out_val_metric = -1.0

                    outdict_computed_metrics[in_thres_value].append(out_val_metric)
                # endfor

                out_computed_metrics_allcases[i,j,:] = outdict_computed_metrics[in_thres_value]
                progressbar.update(1)
            # endfor
        # *******************************************************************************


        # write out computed metrics in file
        out_results_filename = outfilename_metrics_eachcase % (key_casename)
        out_results_filename = join_path_names(output_files_path, out_results_filename)
        if is_exist_file(out_results_filename):
            fout = open(out_results_filename, 'a')
        else:
            fout = open(out_results_filename, 'w')
            strheader = ', '.join(['/thres/'] + ['/%s/' % (key) for key in list_metrics_compute.keys()]) + '\n'
            fout.write(strheader)

        for (in_thres, outlist_computed_metrics) in outdict_computed_metrics.items():
            list_outdata = ['%0.6f' % (in_thres)] + ['%0.6f' % (elem) for elem in outlist_computed_metrics]
            strdata = ', '.join(list_outdata) + '\n'
            fout.write(strdata)
        # endfor
        fout.close()
    # endfor


    # Compute global metrics as mean over all files
    out_mean_allcases_computed_metrics = np.mean(out_computed_metrics_allcases, axis=0)

    out_filename = join_path_names(output_files_path, outfilename_metrics_meanall)
    if is_exist_file(out_filename):
        fout = open(out_filename, 'a')
    else:
        fout = open(out_filename, 'w')
        strheader = ', '.join(['/thres/'] + ['/%s/' % (key) for key in list_metrics_compute.keys()]) + '\n'
        fout.write(strheader)

    for i, in_thres in enumerate(inlist_thresholds):
        list_outdata = ['%0.6f' % (in_thres)] + ['%0.6f' % (elem) for elem in out_mean_allcases_computed_metrics[i]]
        strdata = ', '.join(list_outdata) + '\n'
        fout.write(strdata)
    # endfor
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('input_posteriors_dir', type=str)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--list_type_metrics_ROC_curve', type=str2list_string, default=LIST_TYPE_METRICS_ROC_CURVE)
    parser.add_argument('--is_remove_trachea_calc_metrics', type=str2bool, default=IS_REMOVE_TRACHEA_CALC_METRICS)
    parser.add_argument('--is_connected_masks', type=str2bool, default=False)
    parser.add_argument('--name_input_reference_masks_relpath', type=str, default=NAME_RAW_LABELS_RELPATH)
    parser.add_argument('--name_input_coarse_airways_relpath', type=str, default=NAME_RAW_COARSEAIRWAYS_RELPATH)
    parser.add_argument('--name_input_reference_centrelines_relpath', type=str, default=NAME_RAW_CENTRELINES_RELPATH)
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = args.input_posteriors_dir

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)
