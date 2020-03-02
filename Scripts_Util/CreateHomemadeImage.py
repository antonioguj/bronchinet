#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.Constants import *
from Common.WorkDirsManager import *
from DataLoaders.FileReaders import *
from OperationImages.OperationImages import *
from OperationImages.OperationMasks import *
import argparse



def main(args):

    if args == 'sphere':
        # *********************************************
        print("Create a sphere...")
        coords0 = (128, 128, 128)
        radius = 10

        def wrapfun_sphere_image(in_array):
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

            out_array[indexes_sphere] = 1
            return out_array

        fun_operation = wrapfun_sphere_image
        # *********************************************

    else:
        raise Exception("ERROR. type image \'%s\' not found... EXIT" %(args.type))



    # Create image
    out_array_shape = (256, 256, 256)
    out_array = np.zeros(out_array_shape)

    out_array = fun_operation(out_array)

    # write out image
    print("Write out image: %s..." %(args.outputfile))
    FileReader.writeImageArray(args.outputfile, out_array)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('outputfile', type=str)
    parser.add_argument('--type', type=str, default='None')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)