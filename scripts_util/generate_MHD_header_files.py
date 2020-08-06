
from common.functionutil import *
from dataloaders.imagefilereader import ImageFileReader
import argparse



def main(args):

    names_output_files = lambda in_name: basename_file_noext(in_name) + '.mhd'

    list_input_files = list_files_dir(args.input_dir)
    makedir(args.output_dir)


    for in_file in list_input_files:
        print("\nInput: \'%s\'..." % (basename(in_file)))

        in_image_size  = ImageFileReader.get_image_size(in_file)
        out_image_size = np.roll(in_image_size, 2)

        ndims = len(out_image_size)
        str_out_image_size = ' '.join([str(elem) for elem in out_image_size])
        name_raw_file = basename_file_noext(in_file) + '.raw'

        #contruct mhd header file
        str_info  = 'ObjectType = Image\n'
        str_info += 'NDims = %s\n' %(ndims)
        str_info += 'BinaryData = True\n'
        str_info += 'DimSize = %s\n' %(str_out_image_size)
        str_info += 'ElementType = MET_UCHAR\n'
        str_info += 'ElementDataFile = %s' %(name_raw_file)

        out_file = join_path_names(args.output_dir, names_output_files(in_file))
        print("Output: \'%s\'..." % (basename(out_file)))

        fout = open(out_file, 'w')
        fout.write(str_info)
        fout.close()
    # endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)