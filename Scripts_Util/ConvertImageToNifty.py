#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.FunctionsUtil import *
from DataLoaders.FileReaders import *
import argparse
import sys

#bin_dicom2nifti = '/home/antonio/Libraries/C3d/bin/c3d'
bin_dicom2nifti = '/home/antonio/Libraries/mricron_dcm2niix/dcm2niix'
bin_dicom2nifti_auxdecomp = 'dcmdjpeg'
bin_hr22nifti   = '/home/antonio/Codes/Silas_repository/image-feature-extraction/build/tools/ConvertHR2'



def main(args):
    # ---------- SETTINGS ----------
    InputPath = args.inputdir
    OutputPath = args.outputdir
    namesOutputFiles = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
    # ---------- SETTINGS ----------

    listInputFiles = findFilesDirAndCheck(InputPath)
    makedir(OutputPath)

    files_extension = filenameextension(listInputFiles[0])
    if files_extension == '.dcm':
        files_type = 'dicom'
        tmpfile_template = lambda in_name: filenamenoextension(in_name) + '_dec.dcm'
        tmpsubdir = joinpathnames(InputPath, 'tmp')
        makedir(tmpsubdir)

    elif files_extension == '.hr2':
        files_type = 'hr2'
    else:
        message = 'Extension file \'%s\' not known...' %(files_extension)
        CatchErrorException(message)



    for in_file in listInputFiles:
        print("\nInput: \'%s\'..." %(basename(in_file)))

        out_file = joinpathnames(OutputPath, namesOutputFiles(in_file))
        print("Output: \'%s\'..." % (basename(out_file)))

        if files_type == 'dicom':
            case_file = basename(in_file)
            in_tmp_file = joinpathnames(tmpsubdir, tmpfile_template(in_file))

            # 1st step: decompress input dicom file
            command_string = bin_dicom2nifti_auxdecomp + ' ' + in_file + ' ' + in_tmp_file
            print("%s" % (command_string))
            os.system(command_string)

            # 2nd step: convert decompressed dicom
            command_string = bin_dicom2nifti + ' -o ' + OutputPath + ' -f ' + case_file + ' -z y ' + in_tmp_file
            print("%s" % (command_string))
            os.system(command_string)

            # remove tmp input file and aux. .json file
            out_json_file = joinpathnames(OutputPath, filenamenoextension(out_file) + '.json')
            removefile(in_tmp_file)
            removefile(out_json_file)

            # 3rd step: fix dims of output nifti image and header affine info
            # (THE OUTPUT NIFTI BY THE TOOL dcm2niix HAVE ONE DIMENSION FLIPPED)
            out_image_array = FileReader.getImageArray(out_file)
            out_image_array = NIFTIreader.fixDimsImageArray_fromDicom2niix(out_image_array)

            img_header_affine = NIFTIreader.getImageHeaderInfo(out_file)
            image_position  = DICOMreader.getImagePosition(in_file)
            img_header_affine[1, -1] = image_position[1]
            img_header_affine = NIFTIreader.fixDimsImageAffineMatrix_fromDicom2niix(img_header_affine)

            print("Fix dims of output nifti: \'%s\', with dims: \'%s\'" %(out_file, out_image_array.shape))
            FileReader.writeImageArray(out_file, out_image_array, img_header_info=img_header_affine)

        elif files_type == 'hr2':
            command_string = bin_hr22nifti + ' ' + in_file + ' ' + out_file
            print("%s" % (command_string))
            os.system(command_string)
    #endfor

    if files_type == 'dicom':
        removedir(tmpsubdir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)