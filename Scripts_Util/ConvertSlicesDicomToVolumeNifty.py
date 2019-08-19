#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from DataLoaders.FileReaders import *
import re
import argparse



def main(args):
    # ---------- SETTINGS ----------
    InputPath = args.inputdir
    OutputPath = args.outputdir

    namesInputFiles = '*.dcm'
    namesOutputFiles = lambda in_name: filenamenoextension(in_name) + '.nii.gz'

    patternInputFilename = '[0-9]*.dcm'
    # ---------- SETTINGS ----------

    func_indexSliceFilename  = lambda name: int(re.search(patternInputFilename, name).group(0).replace('.dcm',''))
    func_sortedListByIndexes = lambda list: sorted(list, key=lambda elem: func_indexSliceFilename(elem))

    listInputSubDirs = findFilesDirAndCheck(InputPath,'*')
    makedir(OutputPath)


    for in_subdir in listInputSubDirs:
        print("\nInput: \'%s\'..." % (basename(in_subdir)))

        list_input_slices_files = findFilesDirAndCheck(in_subdir, namesInputFiles)
        list_input_slices_files = func_sortedListByIndexes(list_input_slices_files)

        num_slices = len(list_input_slices_files)
        size_slice = FileReader.getImageSize(list_input_slices_files[0])[1:]

        # create new nifty volume
        out_image_shape = [num_slices] + list(size_slice)
        out_image_array = np.zeros(out_image_shape, dtype=np.float32)

        for i, in_slice_file in enumerate(list_input_slices_files):
            #read slice and stack up in volume array

            in_slice_array = FileReader.getImageArray(in_slice_file)
            out_image_array[i] = np.squeeze(in_slice_array, axis=0)
        #endfor

        out_file = joinpathnames(OutputPath, namesOutputFiles(in_subdir))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_file), str(out_image_array.shape)))

        FileReader.writeImageArray(out_file, out_image_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)