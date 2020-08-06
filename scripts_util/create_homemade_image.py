
from common.functionutil import *
from dataloaders.imagefilereader import ImageFileReader
import argparse



def main(args):

    if args.type == 'sphere':
        # *********************************************
        print("Create a sphere...")
        coords0 = (128, 128, 128)
        radius = 10

        def wrapfun_sphere_image():
            (z_min, z_max) = (coords0[0] - radius, coords0[0] + radius)
            (x_min, x_max) = (coords0[1] - radius, coords0[1] + radius)
            (y_min, y_max) = (coords0[2] - radius, coords0[2] + radius)

            indexZ_sphere = []
            indexX_sphere = []
            indexY_sphere = []
            for iz in range(z_min,z_max,1):
                for ix in range(x_min,x_max,1):
                    for iy in range(y_min,y_max,1):
                        dist = np.sqrt((iz-coords0[0])**2 + (ix-coords0[1])**2 + (iy-coords0[2])**2)
                        if (dist<radius):
                            indexZ_sphere.append(iz)
                            indexX_sphere.append(ix)
                            indexY_sphere.append(iy)
                    #endfor
                #endfor
            #endfor
            indexes_sphere = [indexZ_sphere, indexX_sphere, indexY_sphere]

            out_image[indexes_sphere] = 1
            return out_image

        fun_operation = wrapfun_sphere_image
        # *********************************************

    else:
        raise Exception("ERROR. type image \'%s\' not found... EXIT" %(args.type))


    # Create image
    out_shape_image = (256, 256, 256)
    out_image = np.zeros(out_shape_image)

    out_image = fun_operation(out_image)

    # write out image
    print("Write out image: %s..." %(args.output_file))
    ImageFileReader.write_image(args.output_file, out_image)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', type=str)
    parser.add_argument('--type', type=str, default='None')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)