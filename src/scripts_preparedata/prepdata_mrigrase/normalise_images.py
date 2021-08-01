
import numpy as np
import argparse

from common.functionutil import makedir, join_path_names, basename, list_files_dir
from dataloaders.imagefilereader import ImageFileReader


def main(args):

    list_input_files = list_files_dir(args.inputdir)

    makedir(args.outputdir)

    out_min_val = args.min_value
    out_max_val = args.max_value

    for i, in_file in enumerate(list_input_files):
        print("\nInput: \'%s\'..." % (basename(in_file)))

        in_image = ImageFileReader.get_image(in_file)

        inout_metadata = ImageFileReader.get_image_metadata_info(in_file)

        in_min_val = np.min(in_image)
        in_max_val = np.max(in_image)

        out_image = (out_max_val - out_min_val) * (in_image - in_min_val) / (in_max_val - in_min_val) + out_min_val

        if args.is_out_integer:
            out_image = out_image.astype(np.uint16)

        out_filename = join_path_names(args.outputdir, basename(in_file))
        print("Output: \'%s\'..." % (basename(out_filename)))

        ImageFileReader.write_image(out_filename, out_image, metadata=inout_metadata)
    # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    parser.add_argument('--min_value', type=float, default=0.0)
    parser.add_argument('--max_value', type=float, default=256.0)
    parser.add_argument('--is_out_integer', type=bool, default=False)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
