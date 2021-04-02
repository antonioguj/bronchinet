
from typing import Tuple
import numpy as np
import argparse

from dataloaders.imagefilereader import ImageFileReader


def create_sphere_in_image(inout_image: np.ndarray,
                           coords0: Tuple[int, int, int] = None,
                           radius: float = 50
                           ) -> np.ndarray:
    if not coords0:
        coords0 = (int(inout_image.shape[0]/2), int(inout_image.shape[1]/2), int(inout_image.shape[2]/2))

    (z_min, z_max) = (coords0[0] - radius, coords0[0] + radius)
    (x_min, x_max) = (coords0[1] - radius, coords0[1] + radius)
    (y_min, y_max) = (coords0[2] - radius, coords0[2] + radius)

    indexes_z_sphere = []
    indexes_x_sphere = []
    indexes_y_sphere = []
    for iz in range(z_min, z_max, 1):
        for ix in range(x_min, x_max, 1):
            for iy in range(y_min, y_max, 1):
                dist = np.sqrt((iz - coords0[0]) ** 2 + (ix - coords0[1]) ** 2 + (iy - coords0[2]) ** 2)
                if (dist < radius):
                    indexes_z_sphere.append(iz)
                    indexes_x_sphere.append(ix)
                    indexes_y_sphere.append(iy)
            # endfor
        # endfor
    # endfor
    indexes_sphere = [indexes_z_sphere, indexes_x_sphere, indexes_y_sphere]

    inout_image[indexes_sphere] = 1
    return inout_image


def create_cuboid_in_image(inout_image: np.ndarray,
                           coords0: Tuple[int, int, int] = None,
                           halfsides: Tuple[int, int, int] = (50, 50, 50)
                           ) -> np.ndarray:
    if not coords0:
        coords0 = (int(inout_image.shape[0] / 2), int(inout_image.shape[1] / 2), int(inout_image.shape[2] / 2))

    (z_min, z_max) = (coords0[0] - halfsides[0], coords0[0] + halfsides[0])
    (x_min, x_max) = (coords0[1] - halfsides[1], coords0[1] + halfsides[1])
    (y_min, y_max) = (coords0[2] - halfsides[2], coords0[2] + halfsides[2])

    indexes_z_cuboid = []
    indexes_x_cuboid = []
    indexes_y_cuboid = []
    for iz in range(z_min, z_max, 1):
        for ix in range(x_min, x_max, 1):
            for iy in range(y_min, y_max, 1):
                indexes_z_cuboid.append(iz)
                indexes_x_cuboid.append(ix)
                indexes_y_cuboid.append(iy)
            # endfor
        # endfor
    # endfor
    indexes_cuboid = [indexes_z_cuboid, indexes_y_cuboid, indexes_y_cuboid]

    inout_image[indexes_cuboid] = 1
    return inout_image


def main(args):

    # Create image
    out_shape_image = (288, 288, 96)

    out_image = np.zeros(out_shape_image)
    out_image = create_sphere_in_image(out_image, radius=50)
    # out_image = create_cuboid_in_image(out_image, halfsides=(72, 72, 24))

    # write out image
    print("Write out image: %s..." % (args.output_file))
    ImageFileReader.write_image(args.output_file, out_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
