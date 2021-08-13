
import numpy as np
# from scipy.stats import zscore
import argparse

from common.functionutil import makedir, join_path_names, basename, list_files_dir
from dataloaders.imagefilereader import ImageFileReader


def main(args):

    list_input_files = list_files_dir(args.inputdir)

    makedir(args.outputdir)

    type_normalise = 'minmax-robust'   #'minmax', 'zscore'
    out_min_val = 0.0
    out_max_val = 255.0

    for i, in_file in enumerate(list_input_files):
        print("\nInput: \'%s\'..." % (basename(in_file)))

        inout_image = ImageFileReader.get_image(in_file)

        inout_metadata = ImageFileReader.get_image_metadata_info(in_file)

        if type_normalise == 'minmax':
            in_min_val = np.min(inout_image)
            in_max_val = np.max(inout_image)
            inout_image \
                = (out_max_val - out_min_val) * (inout_image - in_min_val) / (in_max_val - in_min_val) + out_min_val

        elif type_normalise == 'minmax-robust':
            in_min_val = np.percentile(inout_image, 1)
            in_max_val = np.percentile(inout_image, 99)
            inout_image \
                = (out_max_val - out_min_val) * (inout_image - in_min_val) / (in_max_val - in_min_val) + out_min_val

        elif type_normalise == 'zscore':
            in_mean_val = np.mean(inout_image)
            in_std_val = np.std(inout_image)
            inout_image = (inout_image - in_mean_val) / in_std_val

        else:
            print("ERROR: type normalised not valid...")
            exit(0)

        if args.is_out_integer:
            inout_image = inout_image.astype(np.uint16)

        print('New Min value: %s...' % (np.min(inout_image)))
        print('New Max value: %s...' % (np.max(inout_image)))

        out_filename = basename(in_file).replace('.nii.gz', '_normal.nii.gz')
        out_filename = join_path_names(args.outputdir, out_filename)
        print("Output: \'%s\'..." % (basename(out_filename)))

        ImageFileReader.write_image(out_filename, inout_image, metadata=inout_metadata)
    # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    parser.add_argument('--is_out_integer', type=bool, default=False)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
