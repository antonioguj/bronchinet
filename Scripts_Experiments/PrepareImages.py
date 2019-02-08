#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.Constants import *
from Common.WorkDirsManager import *
from DataLoaders.FileReaders import *
from Preprocessing.OperationImages import *
from Preprocessing.OperationMasks import *
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameInputImagesRelPath = 'RawImages'
    nameOutputImagesRelPath = 'Images_Rescaled_0.6x0.6x0.6_Full'

    nameInputImagesFiles = '*.dcm'
    nameOutputImagesFiles = lambda in_name: filenamenoextension(in_name) + '.nii.gz'

    nameRescaleFactors = 'rescaleFactors_images_0.6x0.6x0.6.npy'
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath = workDirsManager.getNameBaseDataPath()
    InputImagesPath = WorkDirsManager.getNameExistPath(BaseDataPath, nameInputImagesRelPath)
    OutputImagesPath = WorkDirsManager.getNameNewPath(BaseDataPath, nameOutputImagesRelPath)

    listInputImagesFiles = findFilesDirAndCheck(InputImagesPath, nameInputImagesFiles)

    if (args.rescaleImages):
        dict_rescaleFactors = readDictionary(joinpathnames(BaseDataPath, nameRescaleFactors))



    for i, in_image_file in enumerate(listInputImagesFiles):
        print("\nInput: \'%s\'..." %(basename(in_image_file)))

        image_array = FileReader.getImageArray(in_image_file)


        if (args.rescaleImages):
            rescale_factor = dict_rescaleFactors[filenamenoextension(in_image_file)]
            print("Rescale image with a factor: \'%s\'..." %(str(rescale_factor)))

            image_array = RescaleImages.compute3D(image_array, rescale_factor)
            print("Final dims: %s..." %(str(image_array.shape)))


        out_file = joinpathnames(OutputImagesPath, nameOutputImagesFiles(in_image_file))
        print("Output: \'%s\', of dims \'%s\'..." %(basename(out_file), str(image_array.shape)))

        FileReader.writeImageArray(out_file, image_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--rescaleImages', type=str2bool, default=RESCALEIMAGES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
