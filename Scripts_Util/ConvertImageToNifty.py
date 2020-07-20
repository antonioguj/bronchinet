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

bin_dicom2nifti = '/home/antonio/Libraries/mricron_dcm2niix/dcm2niix'
bin_dicom2nifti_auxdecomp = 'dcmdjpeg'
bin_hr22nifti   = '/home/antonio/Codes/Silas_repository/image-feature-extraction/build/tools/ConvertHR2'



def main(args):

    namesOutputFiles = lambda in_name: basenameNoextension(in_name) + '.nii.gz'

    list_input_files = findFilesDirAndCheck(args.inputdir)
    makedir(args.outputdir)


    files_extension = fileextension(list_input_files[0])
    if files_extension == '.dcm':
        files_type       = 'dicom'
        tmpfile_template = lambda in_name: basenameNoextension(in_name) + '_dec.dcm'
        tmpsubdir        = joinpathnames(args.inputdir, 'tmp')
        makedir(tmpsubdir)

        if not isExistexec(bin_dicom2nifti):
            message = 'Executable to convert dicom to nifti not found in: %s' %(bin_dicom2nifti)
            CatchErrorException(message)

        if not isExistshexec(bin_dicom2nifti_auxdecomp):
            message = 'Executable to decompress dicom not found in: %s' %(bin_dicom2nifti_auxdecomp)
            CatchErrorException(message)


    elif files_extension == '.hr2':
        files_type = 'hr2'

    elif files_extension == '.mhd':
        files_type = 'mhd'
        if not args.inputRefdir:
            message = 'need to set argument \'inputRefdir\''
            CatchErrorException(message)

        list_input_files = findFilesDirAndCheck(args.inputdir, '*.mhd')
        list_refer_files = findFilesDirAndCheck(args.inputRefdir)
        prefixPatternInputFiles = getFilePrefixPattern(list_refer_files[0])

        if not isExistexec(bin_hr22nifti):
            message = 'Executable to convert hr2 to nifti not found in: %s' %(bin_hr22nifti)
            CatchErrorException(message)

    else:
        message = 'Extension file \'%s\' not known...' %(files_extension)
        CatchErrorException(message)



    for in_file in list_input_files:
        print("\nInput: \'%s\'..." %(basename(in_file)))

        out_file = joinpathnames(args.outputdir, namesOutputFiles(in_file))
        print("Output: \'%s\'..." % (basename(out_file)))

        if files_type == 'dicom':
            case_file   = basename(in_file)
            in_tmp_file = joinpathnames(tmpsubdir, tmpfile_template(in_file))

            # 1st step: decompress input dicom file
            command_string = bin_dicom2nifti_auxdecomp + ' ' + in_file + ' ' + in_tmp_file
            print("%s" % (command_string))
            os.system(command_string)


            # 2nd step: convert decompressed dicom
            command_string = bin_dicom2nifti + ' -o ' + args.outputdir + ' -f ' + case_file + ' -z y ' + in_tmp_file
            print("%s" % (command_string))
            os.system(command_string)

            # remove tmp input file and aux. .json file
            out_json_file = joinpathnames(args.outputdir, basenameNoextension(out_file) + '.json')
            removefile(in_tmp_file)
            removefile(out_json_file)


            # 3rd step: fix dims of output nifti image and header affine info
            # (THE OUTPUT NIFTI BY THE TOOL dcm2niix HAVE ONE DIMENSION FLIPPED)
            out_image_array = FileReader.get_image_array(out_file)
            out_image_array = NIFTIreader.fix_dims_image_array_from_dicom2niix(out_image_array)

            metadata_affine = NIFTIreader.get_image_metadata_info(out_file)
            image_position  = DICOMreader.get_image_position(in_file)
            metadata_affine[1, -1] = image_position[1]
            metadata_affine = NIFTIreader.fix_dims_image_affine_matrix_from_dicom2niix(metadata_affine)

            print("Fix dims of output nifti: \'%s\', with dims: \'%s\'" %(out_file, out_image_array.shape))

            FileReader.write_image_array(out_file, out_image_array, metadata=metadata_affine)


        elif files_type == 'hr2':
            command_string = bin_hr22nifti + ' ' + in_file + ' ' + out_file
            print("%s" % (command_string))
            os.system(command_string)

        elif files_type == 'mhd':
            inout_array = FileReader.get_image_array(in_file)

            in_refer_file = findFileWithSamePrefixPattern(basename(in_file), list_refer_files,
                                                          prefix_pattern=prefixPatternInputFiles)
            in_metadata   = FileReader.get_image_metadata_info(in_refer_file)
            print("Metadata from file: \'%s\'..." % (basename(in_refer_file)))

            FileReader.write_image_array(out_file, inout_array, metadata=in_metadata)
    #endfor

    if files_type == 'dicom':
        removedir(tmpsubdir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    parser.add_argument('--inputRefdir', type=str, default=None)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)