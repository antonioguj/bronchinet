
import argparse
import os

from common.functionutil import makedir, removefile, removedir, join_path_names, is_exist_exec, is_exists_hexec, \
    list_files_dir, basename, basename_filenoext, fileextension, get_regex_pattern_filename, \
    find_file_inlist_with_pattern
from common.exceptionmanager import catch_error_exception
from dataloaders.imagefilereader import ImageFileReader, NiftiReader, DicomReader

BIN_DICOM2NIFTI = '/home/antonio/Libraries/mricron_dcm2niix/dcm2niix'
BIN_DECOMPDICOM = 'dcmdjpeg'
BIN_HR22NIFTI = '/home/antonio/Libraries/image-feature-extraction/build/tools/ConvertHR2'


def main(args):

    def names_output_files(in_name: str):
        return basename_filenoext(in_name) + '.nii.gz'

    list_input_files = list_files_dir(args.input_dir)
    makedir(args.output_dir)

    files_extension = fileextension(list_input_files[0])
    if files_extension == '.dcm':
        files_type = 'dicom'

        def tmpfile_template(in_name: str):
            return basename_filenoext(in_name) + '_dec.dcm'

        tmpsubdir = join_path_names(args.input_dir, 'tmp')
        makedir(tmpsubdir)

        if not is_exist_exec(BIN_DICOM2NIFTI):
            message = 'Executable to convert dicom to nifti not found in: %s' % (BIN_DICOM2NIFTI)
            catch_error_exception(message)

        if not is_exists_hexec(BIN_DECOMPDICOM):
            message = 'Executable to decompress dicom not found in: %s' % (BIN_DECOMPDICOM)
            catch_error_exception(message)

    elif files_extension == '.hr2':
        files_type = 'hr2'

    elif files_extension == '.mhd':
        files_type = 'mhd'
        if not args.input_refdir:
            message = 'need to set argument \'input_refdir\''
            catch_error_exception(message)

        list_input_files = list_files_dir(args.input_dir, '*.mhd')
        list_reference_files = list_files_dir(args.input_refdir)
        pattern_search_infiles = get_regex_pattern_filename(list_reference_files[0])

        if not is_exist_exec(BIN_HR22NIFTI):
            message = 'Executable to convert hr2 to nifti not found in: %s' % (BIN_HR22NIFTI)
            catch_error_exception(message)
    else:
        message = 'Extension file \'%s\' not known...' % (files_extension)
        catch_error_exception(message)

    # ******************************

    for in_file in list_input_files:
        print("\nInput: \'%s\'..." % (basename(in_file)))

        out_file = join_path_names(args.output_dir, names_output_files(in_file))
        print("Output: \'%s\'..." % (basename(out_file)))

        if files_type == 'dicom':
            case_file = basename(in_file)
            in_tmp_file = join_path_names(tmpsubdir, tmpfile_template(in_file))

            # 1st step: decompress input dicom file
            command_string = BIN_DECOMPDICOM + ' ' + in_file + ' ' + in_tmp_file
            print("%s" % (command_string))
            os.system(command_string)

            # 2nd step: convert decompressed dicom
            command_string = BIN_DICOM2NIFTI + ' -o ' + args.output_dir + ' -f ' + case_file + ' -z y ' + in_tmp_file
            print("%s" % (command_string))
            os.system(command_string)

            # remove tmp input file and aux. .json file
            out_json_file = join_path_names(args.output_dir, basename_filenoext(out_file) + '.json')
            removefile(in_tmp_file)
            removefile(out_json_file)

            # 3rd step: fix dims of output nifti image and header affine info
            # (THE OUTPUT NIFTI BY THE TOOL dcm2niix HAVE ONE DIMENSION FLIPPED)
            out_image = ImageFileReader.get_image(out_file)
            out_image = NiftiReader.fix_dims_image_from_dicom2niix(out_image)

            metadata_affine = NiftiReader.get_image_metadata_info(out_file)
            image_position = DicomReader.get_image_position(in_file)
            metadata_affine[1, -1] = image_position[1]
            metadata_affine = NiftiReader.fix_dims_image_affine_matrix_from_dicom2niix(metadata_affine)

            print("Fix dims of output nifti: \'%s\', with dims: \'%s\'" % (out_file, out_image.shape))

            ImageFileReader.write_image(out_file, out_image, metadata=metadata_affine)

        elif files_type == 'hr2':
            command_string = BIN_HR22NIFTI + ' ' + in_file + ' ' + out_file
            print("%s" % (command_string))
            os.system(command_string)

        elif files_type == 'mhd':
            inout_image = ImageFileReader.get_image(in_file)

            in_reference_file = find_file_inlist_with_pattern(basename(in_file), list_reference_files,
                                                              pattern_search=pattern_search_infiles)
            in_metadata = ImageFileReader.get_image_metadata_info(in_reference_file)
            print("Metadata from file: \'%s\'..." % (basename(in_reference_file)))

            ImageFileReader.write_image(out_file, inout_image, metadata=in_metadata)
    # endfor

    if files_type == 'dicom':
        removedir(tmpsubdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--input_refdir', type=str, default=None)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
