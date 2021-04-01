
from common.functionutil import *
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.boundingboxes import BoundingBoxes
from imageoperators.imageoperator import *
import argparse


def compute_test_range_boundbox(in_shape_fullimage, in_shape_cropimage, alpha_relax=0.5, z_min_top=0, z_numtest=1):
    test_range_bounding_boxes = np.zeros((3, 2), dtype=np.int)
    y_0 = (1.0 - alpha_relax) * (int(np.float(in_shape_fullimage[1]) / 2) - int(np.ceil(np.float(in_shape_cropimage[1]) / 2)))
    x_0 = (1.0 - alpha_relax) * (int(np.float(in_shape_fullimage[2]) / 2) - int(np.ceil(np.float(in_shape_cropimage[2]) / 2)))
    z_m = in_shape_fullimage[0] - z_min_top
    z_0 = z_m - in_shape_cropimage[0] - (z_numtest - 1)
    test_range_bounding_boxes[0, 0] = z_0
    test_range_bounding_boxes[0, 1] = z_m
    test_range_bounding_boxes[1, 0] = y_0
    test_range_bounding_boxes[1, 1] = in_shape_fullimage[1] - y_0
    test_range_bounding_boxes[2, 0] = x_0
    test_range_bounding_boxes[2, 1] = in_shape_fullimage[2] - x_0
    return test_range_bounding_boxes

def compute_num_tests_boundbox(test_range_shape_bounding_boxes, in_shape_cropimages):
    num_test_boundbox = test_range_shape_bounding_boxes - in_shape_cropimages + [1, 1, 1]
    num_tests_total = num_test_boundbox[0] * num_test_boundbox[1] * num_test_boundbox[2]
    return (num_test_boundbox, num_tests_total)

def get_limits_test_boundbox(test_range_bounding_boxes, in_size_cropimage, index, option='start_begin'):
    if (option=='start_begin'):
        x0 = test_range_bounding_boxes[0] + index
        xm = x0 + in_size_cropimage
    elif (option=='start_end'):
        xm = test_range_bounding_boxes[1] - index
        x0 = xm - in_size_cropimage
    else:
        return None
    return (x0, xm)



def main(args):
    # ---------- SETTINGS ----------
    #test_range_boundbox = ((16, 352), (109, 433), (45, 460))
    _eps = 1.0e-06
    _alpha_relax = 0.6
    _z_min_top = 15
    _z_numtest = 10
    name_temp_out_res_file   = 'temp_found_boundingBox_vol16.csv'
    # ---------- SETTINGS ----------


    list_input_full_images_files = list_files_dir(args.full_images_dir)
    list_input_crop_images_files = list_files_dir(args.crop_images_dir)

    name_temp_out_res_file = join_path_names(name_temp_out_res_file)
    fout = open(name_temp_out_res_file, 'w')


    dict_found_bounding_boxes = {}

    for in_full_image_file, in_crop_image_file in zip(list_input_full_images_files, list_input_crop_images_files):
        print("\nInput: \'%s\'..." %(basename(in_full_image_file)))
        print("And: \'%s\'..." %(basename(in_crop_image_file)))

        in_full_image = ImageFileReader.get_image(in_full_image_file)
        in_crop_image = ImageFileReader.get_image(in_crop_image_file)
        in_crop_image = FlipImage.compute(in_crop_image, axis=0)

        in_shape_fullimage = np.array(in_full_image.shape)
        in_shape_cropimage = np.array(in_crop_image.shape)
        test_range_bounding_boxes = compute_test_range_boundbox(in_shape_fullimage, in_shape_cropimage,
                                                                alpha_relax=_alpha_relax, z_min_top=_z_min_top, z_numtest=_z_numtest)

        test_range_shape_bounding_boxes = BoundingBoxes.get_size_boundbox(test_range_bounding_boxes)
        if (test_range_shape_bounding_boxes < in_crop_image.shape):
            message = 'size test range of Bounding Boxes than cropped Image: \'%s\' < \'%s\'...' %(test_range_shape_bounding_boxes,
                                                                                                   in_crop_image.shape)
            catch_error_exception(message)
        else:
            test_range_shape_bounding_boxes = np.array(test_range_shape_bounding_boxes)

        (num_test_bounding_boxes, num_tests_total) = compute_num_tests_boundbox(test_range_shape_bounding_boxes, in_crop_image)

        print("size full image: \'%s\'..." %(in_shape_fullimage))
        print("size cropped image: \'%s\'..." %(in_shape_cropimage))
        print("test range bounding boxes: \'%s\'..." %(test_range_bounding_boxes))
        print("size test range bounding boxes: \'%s\'..." %(test_range_shape_bounding_boxes))
        print("num test bounding boxes: \'%s\'..." %(num_test_bounding_boxes))
        print("num tests total: \'%s\'..." %(num_tests_total))


        flag_found_boundbox = False
        min_sum_test_res = 1.0e+10
        found_boundbox = None
        counter = 1
        for k in range(num_test_bounding_boxes[0]):
            (z0, zm) = get_limits_test_boundbox(test_range_bounding_boxes[0], in_shape_cropimage[0], k, option='start_end')
            for j in range(num_test_bounding_boxes[1]):
                (y0, ym) = get_limits_test_boundbox(test_range_bounding_boxes[1], in_shape_cropimage[1], j, option='start_begin')
                for i in range(num_test_bounding_boxes[2]):
                    (x0, xm) = get_limits_test_boundbox(test_range_bounding_boxes[2], in_shape_cropimage[2], i, option='start_begin')
                    #print("test \"%s\" of \"%s\"..." %(counter, num_tests_total))
                    #counter = counter + 1
                    test_bounding_box = ((z0,zm),(y0,ym),(x0,xm))
                    #print("test bounding box: %s..." %(test_bounding_box))
                    test_res_matrix = in_full_image[test_bounding_box[0][0]:test_bounding_box[0][1],
                                                    test_bounding_box[1][0]:test_bounding_box[1][1],
                                                    test_bounding_box[2][0]:test_bounding_box[2][1]] - in_crop_image
                    sum_test_res = np.abs(np.sum(test_res_matrix))
                    if (sum_test_res <_eps):
                        flag_found_boundbox = True
                        min_sum_test_res = 0.0
                        found_boundbox = test_bounding_box
                        break
                    elif (sum_test_res < min_sum_test_res):
                        min_sum_test_res = sum_test_res
                        found_boundbox = test_bounding_box
                if flag_found_boundbox:
                    break
            if flag_found_boundbox:
                break
                # endfor
            # endfor
        # endfor

        if flag_found_boundbox:
            print("SUCCESS: found perfect bounding-box: \'%s\', with null error: \'%s\'..." % (str(found_boundbox), sum_test_res))
            root_cropimage_name = basename_filenoext(in_crop_image_file)
            dict_found_bounding_boxes[root_cropimage_name] = found_boundbox
            message = "%s,\"%s\"\n" %(root_cropimage_name, str(found_boundbox))
            fout.write(message)
        else:
            print("ERROR: not found perfect bounding-box. Closest found is: \'%s\', with error: \'%s\'..." % (str(found_boundbox), min_sum_test_res))
            root_cropimage_name = basename_filenoext(in_crop_image_file)
            dict_found_bounding_boxes[root_cropimage_name] = found_boundbox
            message = "%s,\"%s\" ...NOT PERFECT...\n" % (root_cropimage_name, str(found_boundbox))
            fout.write(message)
    # endfor

    # Save computed bounding-boxes
    save_dictionary(args.found_boundingbox_file, dict_found_bounding_boxes)
    save_dictionary_csv(args.found_boundingbox_file.replace('.npy', '.csv'), dict_found_bounding_boxes)

    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('crop_images_dir', type=str)
    parser.add_argument('full_images_dir', type=str)
    parser.add_argument('--found_boundingbox_file', type=str, default='found_boundingBox_croppedCTinFull.npy')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)