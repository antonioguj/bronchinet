#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.ErrorMessages import *
from Preprocessing.TransformationImages import *
import scipy.ndimage as ndi
from scipy.ndimage import map_coordinates, gaussian_filter
import numpy as np
np.random.seed(2017)



class ElasticDeformationImages(TransformationImages):

    def __init__(self, size_image):
        super(ElasticDeformationImages, self).__init__(size_image)
        self.ndims = len(self.size_image)


    def get_transformed_image(self, X, Y= None, seed= None):
        return NotImplemented

    def get_inverse_transformed_image(self, X, Y= None, seed= None):
        print("Error: Inverse transformation not implemented for 'ElasticDeformationImages'... Exit")
        return 0

    def get_image(self, in_array, in2nd_array= None, index= None, seed= None):
        if in2nd_array is None:
            out_array = self.get_transformed_image(in_array, seed=seed)
            return out_array
        else:
            (out_array, out2nd_array) = self.get_transformed_image(in_array, Y=in2nd_array, seed=seed)
            return (out_array, out2nd_array)



class ElasticDeformationPixelwiseImages(ElasticDeformationImages):
    # implemented by Florian Calvet: florian.calvet@centrale-marseille.fr
    alpha_default = 15
    sigma_default = 3

    def __init__(self, size_image, alpha=alpha_default, sigma=sigma_default):
        self.alpha = alpha
        self.sigma = sigma
        super(ElasticDeformationPixelwiseImages, self).__init__(size_image)


    def get_transformed_image(self, X, Y= None, seed= None):
        if seed is not None:
            np.random.seed(seed)

        shape = X.shape
        if self.ndims == 2:
            xi_dirs = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        elif self.ndims == 3:
            xi_dirs = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        else:
            raise Exception('Error: self.ndims')

        indices = []
        for i in range(self.ndims):
            # originally with random_state.rand * 2 - 1
            dx_i = gaussian_filter(np.random.randn(*shape), self.sigma, mode="constant", cval=0) * self.alpha
            indices.append(xi_dirs[i] + dx_i)
        # endfor

        out_X = map_coordinates(X, indices, order=3).reshape(shape)
        if Y is None:
            return out_X
        else:
            out_Y = map_coordinates(Y, indices, order=0).reshape(shape)
            return (out_X, out_Y)



class ElasticDeformationGridwiseImages(ElasticDeformationImages):
    # implemented by Florian Calvet: florian.calvet@centrale-marseille.fr
    sigma_default = 25
    points_default = 3

    def __init__(self, size_image, sigma=sigma_default, points=points_default):
        self.sigma = sigma
        self.points = points
        super(ElasticDeformationGridwiseImages, self).__init__(size_image)


    def get_transformed_image(self, X, Y= None, seed= None):
        if seed is not None:
            np.random.seed(seed)

        shape = X.shape
        if self.ndims == 2:
            # creates the grid of coordinates of the points of the image (an ndim array per dimension)
            coordinates = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            # creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
            xi = np.meshgrid(np.linspace(0, self.points-1, shape[0]), np.linspace(0, self.points-1, shape[1]), indexing='ij')
            grid = [self.points, self.points]

        elif self.ndims == 3:
            # creates the grid of coordinates of the points of the image (an ndim array per dimension)
            coordinates = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
            # creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
            xi = np.meshgrid(np.linspace(0, self.points-1, shape[0]), np.linspace(0, self.points-1, shape[1]), np.linspace(0, self.points-1, shape[2]), indexing='ij')
            grid = [self.points, self.points, self.points]

        else:
            raise Exception('Error: self.ndims')

        # creates the deformation along each dimension and then add it to the coordinates
        for i in range(len(shape)):
            # creating the displacement at the control points
            yi = np.random.randn(*grid) * self.sigma
            # print(y.shape,coordinates[i].shape) #y and coordinates[i] should be of the same shape otherwise the same displacement is applied to every ?row? of points ?
            y = map_coordinates(yi, xi, order=3).reshape(shape)
            # adding the displacement
            coordinates[i] = np.add(coordinates[i], y)
        #endfor

        out_X = map_coordinates(X, coordinates, order=3).reshape(shape)
        if Y is None:
            return out_X
        else:
            out_Y = map_coordinates(Y, coordinates, order=0).reshape(shape)
            return (out_X, out_Y)