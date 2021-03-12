
from common.functionutil import *
from dataloaders.imagefilereader import ImageFileReader
from collections import OrderedDict
import argparse



def main(args):

    list_input_files_1 = list_files_dir(args.inputdir_1)
    list_input_files_2 = list_files_dir(args.inputdir_2)


    names_files_same_images = []

    for i, in_file_1 in enumerate(list_input_files_1):
        print("\nInput Targer: \'%s\'..." % (in_file_1))

        for j, in_file_2 in enumerate(list_input_files_2):
            print("Try Input: \'%s\'..." % (in_file_2))

            in_image_1 = ImageFileReader.get_image(in_file_1)
            in_image_2 = ImageFileReader.get_image(in_file_2)

            if (in_image_1.shape != in_image_2.shape):
                print('GOOD: Images have different size...')
            else:
                is_images_equal_voxelwise = np.array_equal(in_image_1, in_image_2)

                if is_images_equal_voxelwise:
                    print("BAD: These two images are equal voxelwise...")
                    pair_filenames = (basename(in_file_1), basename(in_file_2))
                    names_files_same_images.append(pair_filenames)
                    break
                else:
                    print('GOOD: Images have same size but are not equal voxelwise...')
        #endfor
    #endfor

    if (len(names_files_same_images) == 0):
        print("\nGOOD: ALL IMAGES IN THE TWO DATASETS ARE DIFFERENT...")
    else:
        print("\nWARNING: Found \'%s\' images that are the same in the two datasets. Names of files: \'%s\'..." %(len(names_files_same_images),
                                                                                                                  names_files_same_images))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir_1', type=str)
    parser.add_argument('inputdir_2', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)