
import numpy as np
import tensorflow as tf
import sys

from dataloaders.imagefilereader import ImageFileReader
from models.metrics import MeanSquaredError, SNR, PSNR, SSIM
from models.keras.metrics import Perceptual


# SETTINGS
#
input_file_1 = '/home/antonio/Data/GRASE2CUBE_Processed/ImagesWorkData/images_proc-01.nii.gz'
input_file_2 = '/home/antonio/Data/GRASE2CUBE_Processed/LabelsWorkData/labels_proc-01.nii.gz'

type_metrics = 'Perceptual'

is_calc_in_slice = False
index_slice = 'middle'
orientation = 'axial'

is_calc_rm_border = True
prop_rm_border = 0.25
#
# --------


image_1 = ImageFileReader.get_image(input_file_1)
image_2 = ImageFileReader.get_image(input_file_2)

if is_calc_in_slice:
    print('ATTENTION: Using the middle 2D slice extracted in \'%s\' dimension...' %(orientation))

    if orientation == 'axial':
        num_slices = image_1.shape[0]
    elif orientation == 'sagital':
        num_slices = image_1.shape[2]
    elif orientation == 'coronal':
        num_slices = image_1.shape[1]
    else:
        print('ERROR: Wrong input \'orientation\'. Only available: [\'axial\', \'sagital\', \'coronal\']...')
        sys.exit(0)

    if index_slice == 'middle':
        index_slice = int(num_slices / 2)

    print('2D slice taken from slice \'%s\'...' %(index_slice))

    if orientation == 'axial':
        image_1 = image_1[index_slice, :, :]
        image_2 = image_2[index_slice, :, :]
    elif orientation == 'sagital':
        image_1 = image_1[:, :, index_slice]
        image_2 = image_2[:, :, index_slice]
    elif orientation == 'coronal':
        image_1 = image_1[:, index_slice, :]
        image_2 = image_2[:, index_slice, :]


if is_calc_rm_border:
    print('ATTENTION: Remove borders of volume, with proportion of \'%s\'...' %(prop_rm_border))

    num_slices_rm_border = tuple([int(s * prop_rm_border / 2) for s in image_1.shape])
    print('Remove num slices in each side in dirs \'%s\'...' %(str(num_slices_rm_border)))

    if len(image_1.shape) == 2:
        image_1 = image_1[num_slices_rm_border[0]: -num_slices_rm_border[0],
                          num_slices_rm_border[1]: -num_slices_rm_border[1]]
        image_2 = image_2[num_slices_rm_border[0]: -num_slices_rm_border[0],
                          num_slices_rm_border[1]: -num_slices_rm_border[1]]

    elif len(image_1.shape) == 3:
        image_1 = image_1[num_slices_rm_border[0]: -num_slices_rm_border[0],
                          num_slices_rm_border[1]: -num_slices_rm_border[1],
                          num_slices_rm_border[2]: -num_slices_rm_border[2]]
        image_2 = image_2[num_slices_rm_border[0]: -num_slices_rm_border[0],
                          num_slices_rm_border[1]: -num_slices_rm_border[1],
                          num_slices_rm_border[2]: -num_slices_rm_border[2]]

    print('Result images of size: \'%s\'...' %(str(image_1.shape)))


if type_metrics == 'MSE':
    metrics = MeanSquaredError()
elif type_metrics == 'SNR':
    metrics = SNR()
elif type_metrics == 'PSNR':
    metrics = PSNR()
elif type_metrics == 'SSIM':
    metrics = SSIM()
elif type_metrics == 'Perceptual':
    metrics = Perceptual(size_image=image_1.shape)
else:
    print('ERROR: Chosen loss function not found...')
    sys.exit(0)

if type_metrics == 'Perceptual':
    image_1 = np.reshape(image_1, (1,) + image_1.shape + (1,))
    image_2 = np.reshape(image_2, (1,) + image_2.shape + (1,))
    
    image_1 = tf.convert_to_tensor(image_1, tf.float32)
    image_2 = tf.convert_to_tensor(image_2, tf.float32)

out_metric = metrics.compute(image_2, image_1)

if type_metrics == 'Perceptual':
    out_val_metric = out_metric.numpy()
else:
    out_val_metric = out_metric

print("\nCalc. metric \'%s\': %s..." % (type_metrics, out_val_metric))
