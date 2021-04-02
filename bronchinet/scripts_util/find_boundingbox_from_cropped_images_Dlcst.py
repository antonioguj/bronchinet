
from typing import Tuple
import numpy as np
import argparse

from common.functionutil import list_files_dir, basename, basename_filenoext, save_dictionary, save_dictionary_csv
from common.exceptionmanager import catch_error_exception
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.boundingboxes import BoundingBoxes
from imageoperators.imageoperator import FlipImage


def compute_test_range_boundbox(in_shape_fullimage: Tuple[int, int, int],
                                in_shape_cropimage: Tuple[int, int, int],
                                alpha_relax: float = 0.5, z_min_top: int = 0, z_numtest: int = 1
                                ) -> np.ndarray:
    test_range_boundboxes = np.zeros((3, 2), dtype=np.int)
    y_0 = (1.0 - alpha_relax) * (int(np.float(in_shape_fullimage[1]) / 2) -
                                 int(np.ceil(np.float(in_shape_cropimage[1]) / 2)))
    x_0 = (1.0 - alpha_relax) * (int(np.float(in_shape_fullimage[2]) / 2) -
                                 int(np.ceil(np.float(in_shape_cropimage[2]) / 2)))
    z_m = in_shape_fullimage[0] - z_min_top
    z_0 = z_m - in_shape_cropimage[0] - (z_numtest - 1)
    test_range_boundboxes[0, 0] = z_0
    test_range_boundboxes[0, 1] = z_m
    test_range_boundboxes[1, 0] = y_0
    test_range_boundboxes[1, 1] = in_shape_fullimage[1] - y_0
    test_range_boundboxes[2, 0] = x_0
    test_range_boundboxes[2, 1] = in_shape_fullimage[2] - x_0
    return test_range_boundboxes


def compute_num_tests_boundbox(test_range_shape_boundboxes: Tuple[int, int, int],
                               in_shape_cropimages: Tuple[int, int, int]
                               ) -> Tuple[int, int]:
    num_test_boundbox = test_range_shape_boundboxes - in_shape_cropimages + [1, 1, 1]
    num_tests_total = num_test_boundbox[0] * num_test_boundbox[1] * num_test_boundbox[2]
    return (num_test_boundbox, num_tests_total)


def get_limits_test_boundbox(test_range_boundboxes: np.ndarray,
                             in_size_cropimage: Tuple[int, int, int],
                             index: int, option: str = 'start_begin'
                             ) -> Tuple[int, int]:
    if (option == 'start_begin'):
        x0 = test_range_boundboxes[0] + index
        xm = x0 + in_size_cropimage
    elif (option == 'start_end'):
        xm = test_range_boundboxes[1] - index
        x0 = xm - in_size_cropimage
    else:
        return None
    return (x0, xm)


def main(args):

    # SETTINGS
    # test_range_boundbox = ((16, 352), (109, 433), (45, 460))
    _eps = 1.0e-06
    _alpha_relax = 0.6
    _z_min_top = 15
    _z_numtest = 10
    name_temp_out_res_file = 'temp_found_boundingBox_vol16.csv'
    # --------

    list_input_full_images_files = list_files_dir(args.full_images_dir)
    list_input_crop_images_files = list_files_dir(args.crop_images_dir)

    fout = open(name_temp_out_res_file, 'w')

    dict_found_boundboxes = {}

    for in_full_image_file, in_crop_image_file in zip(list_input_full_images_files,
                                                      list_input_crop_images_files):
        print("\nInput: \'%s\'..." % (basename(in_full_image_file)))
        print("And: \'%s\'..." % (basename(in_crop_image_file)))

        in_full_image = ImageFileReader.get_image(in_full_image_file)
        in_crop_image = ImageFileReader.get_image(in_crop_image_file)
        in_crop_image = FlipImage.compute(in_crop_image, axis=0)

        in_shape_fullimage = np.array(in_full_image.shape)
        in_shape_cropimage = np.array(in_crop_image.shape)
        test_range_boundboxes = compute_test_range_boundbox(in_shape_fullimage, in_shape_cropimage,
                                                            alpha_relax=_alpha_relax,
                                                            z_min_top=_z_min_top, z_numtest=_z_numtest)

        test_range_shape_boundboxes = BoundingBoxes.get_size_boundbox(test_range_boundboxes)
        if (test_range_shape_boundboxes < in_crop_image.shape):
            message = 'size test range of Bounding Boxes than cropped Image: \'%s\' < \'%s\'...' \
                      % (test_range_shape_boundboxes, in_crop_image.shape)
            catch_error_exception(message)
        else:
            test_range_shape_boundboxes = np.array(test_range_shape_boundboxes)

        (num_test_boundboxes, num_tests_total) = compute_num_tests_boundbox(test_range_shape_boundboxes, in_crop_image)

        print("size full image: \'%s\'..." % (in_shape_fullimage))
        print("size cropped image: \'%s\'..." % (in_shape_cropimage))
        print("test range bounding-boxes: \'%s\'..." % (test_range_boundboxes))
        print("size test range bounding-boxes: \'%s\'..." % (test_range_shape_boundboxes))
        print("num test bounding-boxes: \'%s\'..." % (num_test_boundboxes))
        print("num tests total: \'%s\'..." % (num_tests_total))

        flag_found_boundbox = False
        min_sum_test_res = 1.0e+10
        found_boundbox = None
        # counter = 1
        for k in range(num_test_boundboxes[0]):
            (z0, zm) = get_limits_test_boundbox(test_range_boundboxes[0], in_shape_cropimage[0], k,
                                                option='start_end')
            for j in range(num_test_boundboxes[1]):
                (y0, ym) = get_limits_test_boundbox(test_range_boundboxes[1], in_shape_cropimage[1], j,
                                                    option='start_begin')
                for i in range(num_test_boundboxes[2]):
                    (x0, xm) = get_limits_test_boundbox(test_range_boundboxes[2], in_shape_cropimage[2], i,
                                                        option='start_begin')
                    # print("test \"%s\" of \"%s\"..." % (counter, num_tests_total))
                    # counter = counter + 1
                    test_boundbox = ((z0, zm), (y0, ym), (x0, xm))
                    # print("test bounding-box: %s..." % (test_boundbox))
                    test_res_matrix = in_full_image[test_boundbox[0][0]:test_boundbox[0][1],
                                                    test_boundbox[1][0]:test_boundbox[1][1],
                                                    test_boundbox[2][0]:test_boundbox[2][1]] - in_crop_image
                    sum_test_res = np.abs(np.sum(test_res_matrix))
                    if (sum_test_res < _eps):
                        flag_found_boundbox = True
                        min_sum_test_res = 0.0
                        found_boundbox = test_boundbox
                        break
                    elif (sum_test_res < min_sum_test_res):
                        min_sum_test_res = sum_test_res
                        found_boundbox = test_boundbox
                if flag_found_boundbox:
                    break
            if flag_found_boundbox:
                break
                # endfor
            # endfor
        # endfor

        if flag_found_boundbox:
            print("SUCCESS: found perfect bounding-box: \'%s\', with null error: \'%s\'..."
                  % (str(found_boundbox), sum_test_res))
            root_cropimage_name = basename_filenoext(in_crop_image_file)
            dict_found_boundboxes[root_cropimage_name] = found_boundbox
            message = "%s, \'%s\'\n" % (root_cropimage_name, str(found_boundbox))
            fout.write(message)
        else:
            print("ERROR: not found perfect bounding-box. Closest found is: \'%s\', with error: \'%s\'..."
                  % (str(found_boundbox), min_sum_test_res))
            root_cropimage_name = basename_filenoext(in_crop_image_file)
            dict_found_boundboxes[root_cropimage_name] = found_boundbox
            message = "%s, \'%s\' ...NOT PERFECT...\n" % (root_cropimage_name, str(found_boundbox))
            fout.write(message)
    # endfor

    # Save computed bounding-boxes
    save_dictionary(args.found_boundboxes_file, dict_found_boundboxes)
    save_dictionary_csv(args.found_boundboxes_file.replace('.npy', '.csv'), dict_found_boundboxes)

    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('crop_images_dir', type=str)
    parser.add_argument('full_images_dir', type=str)
    parser.add_argument('--found_boundboxes_file', type=str, default='found_boundingBox_croppedCTinFull.npy')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
