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
from Preprocessing.TransformationRigidImages import *
import scipy.ndimage as ndi
from scipy.ndimage import map_coordinates, gaussian_filter
import numpy as np
np.random.seed(2017)



class TransformElasticDeformImages(TransformationRigidImages):

    def __init__(self, size_image):
        super(TransformElasticDeformImages, self).__init__(size_image)
        self.ndims = len(self.size_image)


    def compute_gendata(self, **kwargs):
        seed = kwargs['seed']
        self.gendata_elastic_deform = self.get_compute_gendata_elastic_deform(seed)
        self.is_compute_gendata = False

    def initialize_gendata(self):
        self.is_compute_gendata = True
        self.transformation_matrix = None

    def get_text_description(self):
        return 'Elastic Deformations of image patches...\n'


    def get_compute_gendata_elastic_deform(self, seed= None):
        return NotImplemented


    def get_transformed_image(self, in_array, is_image_array= False):
        out_array = map_coordinates(in_array, self.gendata_elastic_deform, order=3).reshape(self.size_image)
        return out_array

    def get_inverse_transformed_image(self, in_array, is_image_array= False):
        print("Error: Inverse transformation not implemented for Elastic Deformations... Exit")
        return 0


    def get_image(self, in_array):
        return self.get_transformed_image(in_array)



class TransformElasticDeformPixelwiseImages(TransformElasticDeformImages):
    # implemented by Florian Calvet: florian.calvet@centrale-marseille.fr
    alpha_default = 15
    sigma_default = 3

    def __init__(self, size_image, alpha=alpha_default, sigma=sigma_default):
        self.alpha = alpha
        self.sigma = sigma
        super(TransformElasticDeformPixelwiseImages, self).__init__(size_image)


    def get_compute_gendata_elastic_deform(self, seed= None):
        if seed is not None:
            np.random.seed(seed)

        if self.ndims == 2:
            xi_dirs = np.meshgrid(np.arange(self.size_image[0]),
                                  np.arange(self.size_image[1]), indexing='ij')
        elif self.ndims == 3:
            xi_dirs = np.meshgrid(np.arange(self.size_image[0]), np.arange(self.size_image[1]),
                                  np.arange(self.size_image[2]), indexing='ij')
        else:
            raise Exception('Error: self.ndims')

        indices = []
        for i in range(self.ndims):
            # originally with random_state.rand * 2 - 1
            dx_i = gaussian_filter(np.random.randn(*self.size_image), self.sigma, mode="constant", cval=0) * self.alpha
            indices.append(xi_dirs[i] + dx_i)
        # endfor

        return indices



class TransformElasticDeformGridwiseImages(TransformElasticDeformImages):
    # implemented by Florian Calvet: florian.calvet@centrale-marseille.fr
    sigma_default = 25
    points_default = 3

    def __init__(self, size_image, sigma=sigma_default, points=points_default):
        self.sigma = sigma
        self.points = points
        super(TransformElasticDeformGridwiseImages, self).__init__(size_image)


    def get_compute_gendata_elastic_deform(self, seed= None):
        if seed is not None:
            np.random.seed(seed)

        if self.ndims == 2:
            # creates the grid of coordinates of the points of the image (an ndim array per dimension)
            coordinates = np.meshgrid(np.arange(self.size_image[0]),
                                      np.arange(self.size_image[1]), indexing='ij')
            # creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
            xi = np.meshgrid(np.linspace(0, self.points-1, self.size_image[0]),
                             np.linspace(0, self.points-1, self.size_image[1]), indexing='ij')
            grid = [self.points, self.points]

        elif self.ndims == 3:
            # creates the grid of coordinates of the points of the image (an ndim array per dimension)
            coordinates = np.meshgrid(np.arange(self.size_image[0]),
                                      np.arange(self.size_image[1]),
                                      np.arange(self.size_image[2]), indexing='ij')
            # creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
            xi = np.meshgrid(np.linspace(0, self.points-1, self.size_image[0]),
                             np.linspace(0, self.points-1, self.size_image[1]),
                             np.linspace(0, self.points-1, self.size_image[2]), indexing='ij')
            grid = [self.points, self.points, self.points]

        else:
            raise Exception('Error: self.ndims')

        # creates the deformation along each dimension and then add it to the coordinates
        for i in range(self.ndims):
            # creating the displacement at the control points
            yi = np.random.randn(*grid) * self.sigma
            # print(y.shape,coordinates[i].shape) #y and coordinates[i] should be of the same shape otherwise the same displacement is applied to every ?row? of points ?
            y = map_coordinates(yi, xi, order=3).reshape(self.size_image)
            # adding the displacement
            coordinates[i] = np.add(coordinates[i], y)
        #endfor

        return coordinates