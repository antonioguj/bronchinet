
from common.functionutil import *
from dataloaders.imagefilereader import ImageFileReader
import argparse


def create_sphere_in_image(inout_image: np.ndarray, coords0=None, radius=50) -> np.ndarray:
    if not coords0:
        coords0 = (int(inout_image.shape[0]/2), int(inout_image.shape[1]/2), int(inout_image.shape[2]/2))

    (z_min, z_max) = (coords0[0] - radius, coords0[0] + radius)
    (x_min, x_max) = (coords0[1] - radius, coords0[1] + radius)
    (y_min, y_max) = (coords0[2] - radius, coords0[2] + radius)

    indexZ_sphere = []
    indexX_sphere = []
    indexY_sphere = []
    for iz in range(z_min, z_max, 1):
        for ix in range(x_min, x_max, 1):
            for iy in range(y_min, y_max, 1):
                dist = np.sqrt((iz - coords0[0]) ** 2 + (ix - coords0[1]) ** 2 + (iy - coords0[2]) ** 2)
                if (dist < radius):
                    indexZ_sphere.append(iz)
                    indexX_sphere.append(ix)
                    indexY_sphere.append(iy)
            # endfor
        # endfor
    # endfor
    indexes_sphere = [indexZ_sphere, indexX_sphere, indexY_sphere]

    inout_image[indexes_sphere] = 1
    return inout_image


def create_cuboid_in_image(inout_image: np.ndarray, coords0=None, halfsides=(50, 50, 50)) -> np.ndarray:
    if not coords0:
        coords0 = (int(inout_image.shape[0] / 2), int(inout_image.shape[1] / 2), int(inout_image.shape[2] / 2))

    (z_min, z_max) = (coords0[0] - halfsides[0], coords0[0] + halfsides[0])
    (x_min, x_max) = (coords0[1] - halfsides[1], coords0[1] + halfsides[1])
    (y_min, y_max) = (coords0[2] - halfsides[2], coords0[2] + halfsides[2])

    indexZ_cuboid = []
    indexX_cuboid = []
    indexY_cuboid = []
    for iz in range(z_min, z_max, 1):
        for ix in range(x_min, x_max, 1):
            for iy in range(y_min, y_max, 1):
                indexZ_cuboid.append(iz)
                indexX_cuboid.append(ix)
                indexY_cuboid.append(iy)
            # endfor
        # endfor
    # endfor
    indexes_cuboid = [indexZ_cuboid, indexX_cuboid, indexY_cuboid]

    inout_image[indexes_cuboid] = 1
    return inout_image



def main(args):


    # Create image
    out_shape_image = (256, 256, 256)

    out_image_1 = np.zeros(out_shape_image)
    out_image_1 = create_sphere_in_image(out_image_1, radius=50)
    #out_image_1 = create_cuboid_in_image(out_image_1, halfsides=(50, 50, 50))

    out_image = out_image_1

    # write out image
    print("Write out image: %s..." %(args.output_file))
    ImageFileReader.write_image(args.output_file, out_image)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)
