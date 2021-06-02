
import numpy as np
import argparse

from common.functionutil import makedir, join_path_names, list_files_dir, basename, basename_filenoext
from dataloaders.imagefilereader import ImageFileReader


def main(args):

    def names_output_files(in_name: str):
        return basename_filenoext(in_name) + '.mhd'

    list_input_files = list_files_dir(args.input_dir)
    makedir(args.output_dir)

    for in_file in list_input_files:
        print("\nInput: \'%s\'..." % (basename(in_file)))

        in_image_size = ImageFileReader.get_image_size(in_file)
        out_image_size = np.roll(in_image_size, 2)

        ndims = len(out_image_size)
        str_out_image_size = ' '.join([str(elem) for elem in out_image_size])
        name_raw_file = basename_filenoext(in_file) + '.raw'

        # construct mhd header file
        str_info = 'ObjectType = Image\n'
        str_info += 'NDims = %s\n' % (ndims)
        str_info += 'BinaryData = True\n'
        str_info += 'DimSize = %s\n' % (str_out_image_size)
        str_info += 'ElementType = MET_UCHAR\n'
        str_info += 'ElementDataFile = %s' % (name_raw_file)

        out_file = join_path_names(args.output_dir, names_output_files(in_file))
        print("Output: \'%s\'..." % (basename(out_file)))

        with open(out_file, 'w') as fout:
            fout.write(str_info)
    # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
