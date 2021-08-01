
import numpy as np
import tensorflow as tf
import sys

from dataloaders.imagefilereader import ImageFileReader
from models.keras.metrics import L1, L2, DSSIM, Perceptual


input_file_1 = '/home/antonio/Data/GRASE2CUBE_Processed/ImagesWorkData/images_proc-01.nii.gz'
input_file_2 = '/home/antonio/Data/GRASE2CUBE_Processed/LabelsWorkData/labels_proc-01.nii.gz'

type_metrics = 'DSSIM'


image_1 = ImageFileReader.get_image(input_file_1)
image_2 = ImageFileReader.get_image(input_file_2)

if type_metrics == 'L1':
    metrics = L1()
elif type_metrics == 'L2':
    metrics = L2()
elif type_metrics == 'DSSIM':
    metrics = DSSIM()
elif type_metrics == 'Perceptual':
    metrics = Perceptual(size_image=image_1.shape)
else:
    print('ERROR: Chosen loss function not found...')
    sys.exit(0)

image_1 = np.reshape(image_1, (1,) + image_1.shape + (1,))
image_2 = np.reshape(image_2, (1,) + image_2.shape + (1,))

image_1 = tf.convert_to_tensor(image_1, tf.float32)
image_2 = tf.convert_to_tensor(image_2, tf.float32)

out_loss = metrics.lossfun(image_2, image_1)
out_val_loss = out_loss.numpy()

print("\nCalc. metric \'%s\': %s..." % (type_metrics, out_val_loss))
