
from collections import OrderedDict
import argparse

from common.functionutil import makedir, makelink, join_path_names, basename, basename_filenoext, list_files_dir, \
    save_dictionary, save_dictionary_csv


def main(args):

    name_template_output_images_files = 'images_proc-%0.2i.nii.gz'
    name_template_output_labels_files = 'labels_proc-%0.2i.nii.gz'

    input_images_path = join_path_names(args.indatadir, 'Images/')
    input_labels_path = join_path_names(args.indatadir, 'Labels/')

    output_images_path = join_path_names(args.indatadir, 'ImagesWorkData/')
    output_labels_path = join_path_names(args.indatadir, 'LabelsWorkData/')
    out_reference_keys_file = join_path_names(args.indatadir, 'referenceKeys_procimages.npy')

    list_input_images_files = list_files_dir(input_images_path)
    list_input_labels_files = list_files_dir(input_labels_path)

    if len(list_input_images_files) != len(list_input_labels_files):
        message = 'num files in two lists not equal: \'%s\' != \'%s\'...' \
                  % (len(list_input_images_files), len(list_input_labels_files))
        catch_error_exception(message)

    makedir(output_images_path)
    makedir(output_labels_path)

    # --------------------

    outdict_reference_keys = OrderedDict()

    for ifile, (in_image_file, in_label_file) in enumerate(zip(list_input_images_files,
                                                               list_input_labels_files)):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))
        print("And: \'%s\'..." % (basename(in_label_file)))

        out_image_file = name_template_output_images_files % (ifile + 1)
        out_image_file = join_path_names(output_images_path, out_image_file)

        out_label_file = name_template_output_labels_files % (ifile + 1)
        out_label_file = join_path_names(output_labels_path, out_label_file)

        in_image_file = join_path_names('../', in_image_file)
        print("%s --> %s" % (basename(out_image_file), in_image_file))
        makelink(in_image_file, out_image_file)

        in_label_file = join_path_names('../', in_label_file)
        print("%s --> %s" % (basename(out_label_file), in_label_file))
        makelink(in_label_file, out_label_file)

        outdict_reference_keys[basename_filenoext(out_image_file)] = basename(in_image_file)
    # endfor

    save_dictionary(out_reference_keys_file, outdict_reference_keys)
    save_dictionary_csv(out_reference_keys_file.replace('.npy', '.csv'), outdict_reference_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indatadir', type=str, default='.')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)