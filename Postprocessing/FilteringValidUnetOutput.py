#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.Constants import *
from CommonUtil.ErrorMessages import *
from Preprocessing.BaseImageGenerator import *


class FilteringValidUnetOutput(BaseImageGenerator):

    type_progression_outside_output_Unet = 'quadratic'
    avail_type_progression_outside_output_Unet = ['linear', 'quadratic', 'cubic', 'exponential', 'all_outputs_Unet']


    def __init__(self, size_image, size_valid_outUnet):
        self.size_image         = size_image
        self.size_valid_outUnet = size_valid_outUnet
        try:
            # if len(self.size_valid_outUnet) == 1:
            # a bit sloppy: integer doesn't have attribute 'len'
            dummy = len(self.size_valid_outUnet[0])
            self.size_boundbox_outUnet = self.size_valid_outUnet + [self.size_image]
            self.type_progression = 'all_outputs_Unet'
        except TypeError:
            self.type_progression = self.type_progression_outside_output_Unet
            self.size_boundbox_outUnet = [self.size_valid_outUnet] + [self.size_image]

        self.compute_filter_func_outUnet()

        super(FilteringValidUnetOutput, self).__init__(size_image)


    @staticmethod
    def multiply_matrixes_with_channels(matrix_1_withchannels, matrix_2):
        pass

    @staticmethod
    def compute_tensor_product_2D(a, b):
        return np.einsum('i,j->ij', a, b)

    @staticmethod
    def compute_tensor_product_3D(a, b, c):
        return np.einsum('i,j,k->ijk', a, b, c)

    @staticmethod
    def compute_linear_progression(coord_0, coord_1):
        return np.linspace(0, 1, coord_1 - coord_0)

    @staticmethod
    def compute_quadratic_progression(coord_0, coord_1):
        return np.power(np.linspace(0, 1, coord_1 - coord_0), 2)

    @staticmethod
    def compute_cubic_progression(coord_0, coord_1):
        return np.power(np.linspace(0, 1, coord_1 - coord_0), 3)

    @staticmethod
    def compute_exponential_progression(coord_0, coord_1):
        return (np.exp(np.linspace(0, 1, coord_1 - coord_0)) - 1)/(np.exp(1) - 1)

    def compute_progression_increasing(self, coord_0, coord_1):
        if self.type_progression == 'linear':
            return self.compute_linear_progression(coord_0, coord_1)
        elif self.type_progression == 'quadratic':
            return self.compute_quadratic_progression(coord_0, coord_1)
        elif self.type_progression == 'cubic':
            return self.compute_cubic_progression(coord_0, coord_1)
        elif self.type_progression == 'exponential':
            return self.compute_exponential_progression(coord_0, coord_1)
        elif self.type_progression == 'all_outputs_Unet':
            # assume piecewise quadratic progression
            return self.compute_quadratic_progression(coord_0, coord_1)
        else:
            return 0

    def compute_progression_decreasing(self, coord_0, coord_1):
        return self.compute_progression_increasing(coord_0, coord_1)[::-1]

    @staticmethod
    def get_limits_cropImage(size_image, size_valid_outUnet):
        if (size_image == size_valid_outUnet):
            list_out_aux = [[0] + [s_i] for s_i in size_image]
        else:
            list_out_aux = [[(s_i - s_o) / 2] + [(s_i + s_o) / 2] for (s_i, s_o) in zip(size_image, size_valid_outUnet)]
        # flatten out list of lists and return tuple
        return tuple(reduce(lambda el1, el2: el1 + el2, list_out_aux))

    def fill_flat_interior_boundbox(self, boundbox_inner):
        pass

    def fill_progression_between_two_boundboxes(self, boundbox_in, boundbox_out, prop_val_in=1.0, prop_val_out=0.0):
        pass

    def fill_flat_exterior_boundbox(self, boundbox_outer):
        pass


    def complete_init_data(self, in_array_shape):
        self.num_images = in_array_shape[0]

    def get_num_images(self):
        return self.num_images

    def get_shape_out_array(self, in_array_shape):
        num_images   = in_array_shape[0]
        num_channels = self.get_num_channels_array(in_array_shape[1:0])
        if num_channels:
            return [num_images] + list(self.size_image) + [num_channels]
        else:
            return [num_images] + list(self.size_image)

    def check_correct_dims_image_to_filter(self, in_array_shape):
        if self.is_images_array_without_channels(in_array_shape):
            in_array_size_image = in_array_shape
        else:
            in_array_size_image = in_array_shape[:-1]
        if in_array_size_image == self.size_image:
            return True
        else:
            return False


    def get_filter_func_outUnet_array(self):
        return self.filter_func_outUnet_array

    def compute_filter_func_outUnet(self):

        self.filter_func_outUnet_array = np.zeros(self.size_image, dtype=FORMATPROBABILITYDATA)

        if self.type_progression == 'all_outputs_Unet':
            # set flat probability equal to 'one' inside inner output of Unet
            # set piecewise probability distribution in between bounding boxes corresponding to outputs of Unet up to 'max_depth'
            # in between bounding boxes, assume quadratic distribution in between two values with diff: 1/num_output_Unet

            num_output_Unet = len(self.size_valid_outUnet)

            boundbox_inner_outUnet = self.get_limits_cropImage(self.size_image, self.size_boundbox_outUnet[0])

            self.fill_flat_interior_boundbox(boundbox_inner_outUnet)

            for i in range(num_output_Unet):
                prop_val_in  = 1.0 - i / float(num_output_Unet)
                prop_val_out = 1.0 - (i + 1) / float(num_output_Unet)

                boundbox_inner_outUnet = self.get_limits_cropImage(self.size_image, self.size_boundbox_outUnet[i])
                boundbox_outer_outUnet = self.get_limits_cropImage(self.size_image, self.size_boundbox_outUnet[i+1])

                self.fill_progression_between_two_boundboxes(boundbox_inner_outUnet, boundbox_outer_outUnet, prop_val_in, prop_val_out)
            # endfor
        else:
            # set flat probability equal to 'one' inside output of Unet
            # set probability distribution (linear, quadratic, ...) in between output of Unet and borders of image

            boundbox_inner_outUnet = self.get_limits_cropImage(self.size_image, self.size_boundbox_outUnet[0])
            boundbox_size_image    = self.get_limits_cropImage(self.size_image, self.size_boundbox_outUnet[1])

            self.fill_flat_interior_boundbox(boundbox_inner_outUnet)

            self.fill_progression_between_two_boundboxes(boundbox_inner_outUnet, boundbox_size_image)


    def get_filtered_images_array(self, images_array):

        if self.check_correct_dims_image_to_filter(images_array.shape):
            if self.is_images_array_without_channels(images_array.shape):
                return np.multiply(self.filter_func_outUnet_array, images_array)
            else:
                return self.multiply_matrixes_with_channels(images_array, self.filter_func_outUnet_array)
        else:
            return None


    def get_images_array(self, images_array, index=None, masks_array=None, seed=None):

        out_images_array = self.get_filtered_images_array(images_array)

        if masks_array is None:
            return out_images_array
        else:
            out_masks_array = self.get_filtered_images_array(masks_array)

            return (out_images_array, out_masks_array)

    def compute_images_array_all(self, images_array, masks_array=None, seed_0=None):

        out_array_shape  = self.get_shape_out_array(images_array.shape)
        out_images_array = np.ndarray(out_array_shape, dtype=images_array.dtype)

        if masks_array is None:
            num_images = images_array.shape[0]
            for index in range(num_images):
                out_images_array[index] = self.get_filtered_images_array(images_array[index])
            #endfor

            return out_images_array
        else:
            out_array_shape = self.get_shape_out_array(masks_array.shape)
            out_masks_array = np.ndarray(out_array_shape, dtype=masks_array.dtype)

            num_images = images_array.shape[0]
            for index in range(num_images):
                out_images_array[index] = self.get_filtered_images_array(images_array[index])
                out_masks_array [index] = self.get_filtered_images_array(masks_array [index])
            #endfor

            return (out_images_array, out_masks_array)


class FilteringValidUnetOutput2D(FilteringValidUnetOutput):

    def __init__(self, size_image, size_valid_outUnet):
        super(FilteringValidUnetOutput2D, self).__init__(size_image, size_valid_outUnet)


    @staticmethod
    def multiply_matrixes_with_channels(matrix_1_withchannels, matrix_2):
        return np.einsum('ijk,ij->kij', matrix_1_withchannels, matrix_2)

    def fill_flat_interior_boundbox(self, boundbox_inner):
        # assign probability 'one' inside box
        (x_left, x_right, y_down, y_up) = boundbox_inner

        self.filter_func_outUnet_array[x_left:x_right, y_down:y_up] = 1.0

    def fill_progression_between_two_boundboxes(self, boundbox_in, boundbox_out, prop_val_in=1.0, prop_val_out=0.0):
        # assign probability distribution between 'inner' and 'outer' boundboxes, between values 'prop_output_val_in' and 'prop_output_val_out'

        (x_left_in,  x_right_in,  y_down_in,  y_up_in ) = boundbox_in
        (x_left_out, x_right_out, y_down_out, y_up_out) = boundbox_out

        progression_x_left  = self.compute_progression_increasing(x_left_out, x_left_in  ) * (prop_val_in - prop_val_out) + prop_val_out
        progression_x_right = self.compute_progression_decreasing(x_right_in, x_right_out) * (prop_val_in - prop_val_out) + prop_val_out
        progression_y_down  = self.compute_progression_increasing(y_down_out, y_down_in  ) * (prop_val_in - prop_val_out) + prop_val_out
        progression_y_up    = self.compute_progression_decreasing(y_up_in,    y_up_out   ) * (prop_val_in - prop_val_out) + prop_val_out
        progression_x_middle = np.ones([x_right_in - x_left_in]) * prop_val_in
        progression_y_middle = np.ones([y_up_in    - y_down_in]) * prop_val_in

        # laterals
        self.filter_func_outUnet_array[x_left_out:x_left_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_2D(progression_x_left,   progression_y_middle)
        self.filter_func_outUnet_array[x_right_in:x_right_out, y_down_in:y_up_in   ] = self.compute_tensor_product_2D(progression_x_right,  progression_y_middle)
        self.filter_func_outUnet_array[x_left_in:x_right_in,   y_down_out:y_down_in] = self.compute_tensor_product_2D(progression_x_middle, progression_y_down  )
        self.filter_func_outUnet_array[x_left_in:x_right_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_2D(progression_x_middle, progression_y_up    )
        # corners
        self.filter_func_outUnet_array[x_left_out:x_left_in,   y_down_out:y_down_in] = self.compute_tensor_product_2D(progression_x_left,   progression_y_down  )
        self.filter_func_outUnet_array[x_right_in:x_right_out, y_down_out:y_down_in] = self.compute_tensor_product_2D(progression_x_right,  progression_y_down  )
        self.filter_func_outUnet_array[x_left_out:x_left_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_2D(progression_x_left,   progression_y_up    )
        self.filter_func_outUnet_array[x_right_in:x_right_out, y_up_in:y_up_out    ] = self.compute_tensor_product_2D(progression_x_right,  progression_y_up    )

    def fill_flat_exterior_boundbox(self, boundbox_outer):
        # assign probability 'zero' outside box
        (x_left, x_right, y_down, y_up) = boundbox_outer

        self.filter_func_outUnet_array[0:x_left, :       ] = 0.0
        self.filter_func_outUnet_array[x_right:, :       ] = 0.0
        self.filter_func_outUnet_array[:,        0:y_down] = 0.0
        self.filter_func_outUnet_array[:,        y_up:   ] = 0.0


class FilteringValidUnetOutput3D(FilteringValidUnetOutput):

    def __init__(self, size_image, size_valid_outUnet):
        super(FilteringValidUnetOutput3D, self).__init__(size_image, size_valid_outUnet)


    @staticmethod
    def multiply_matrixes_with_channels(matrix_1_withchannels, matrix_2):
        return np.einsum('ijkl,ijk->ijkl', matrix_1_withchannels, matrix_2)

    def fill_flat_interior_boundbox(self, boundbox_inner):
        # assign probability 'one' inside box
        (z_back, z_front, x_left, x_right, y_down, y_up) = boundbox_inner

        self.filter_func_outUnet_array[z_back:z_front, x_left:x_right, y_down:y_up] = 1.0

    def fill_progression_between_two_boundboxes(self, boundbox_in, boundbox_out, prop_val_in=1.0, prop_val_out=0.0):
        # assign probability distribution between 'inner' and 'outer' boundboxes, between values 'prop_output_val_in' and 'prop_output_val_out'

        (z_back_in,  z_front_in,  x_left_in,  x_right_in,  y_down_in,  y_up_in ) = boundbox_in
        (z_back_out, z_front_out, x_left_out, x_right_out, y_down_out, y_up_out) = boundbox_out

        progression_z_back  = self.compute_progression_increasing(z_back_out, z_back_in  ) * (prop_val_in - prop_val_out) + prop_val_out
        progression_z_front = self.compute_progression_decreasing(z_front_in, z_front_out) * (prop_val_in - prop_val_out) + prop_val_out
        progression_x_left  = self.compute_progression_increasing(x_left_out, x_left_in  ) * (prop_val_in - prop_val_out) + prop_val_out
        progression_x_right = self.compute_progression_decreasing(x_right_in, x_right_out) * (prop_val_in - prop_val_out) + prop_val_out
        progression_y_down  = self.compute_progression_increasing(y_down_out, y_down_in  ) * (prop_val_in - prop_val_out) + prop_val_out
        progression_y_up    = self.compute_progression_decreasing(y_up_in,    y_up_out   ) * (prop_val_in - prop_val_out) + prop_val_out
        progression_z_middle = np.ones([z_front_in - z_back_in]) * prop_val_in
        progression_x_middle = np.ones([x_right_in - x_left_in]) * prop_val_in
        progression_y_middle = np.ones([y_up_in    - y_down_in]) * prop_val_in

        # laterals
        self.filter_func_outUnet_array[z_back_in:z_front_in,   x_left_out:x_left_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_left,   progression_y_middle)
        self.filter_func_outUnet_array[z_back_in:z_front_in,   x_right_in:x_right_out, y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_right,  progression_y_middle)
        self.filter_func_outUnet_array[z_back_in:z_front_in,   x_left_in:x_right_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_middle, progression_x_middle, progression_y_down  )
        self.filter_func_outUnet_array[z_back_in:z_front_in,   x_left_in:x_right_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_middle, progression_y_up    )
        self.filter_func_outUnet_array[z_back_out:z_back_in,   x_left_in:x_right_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_middle, progression_y_middle)
        self.filter_func_outUnet_array[z_front_in:z_front_out, x_left_in:x_right_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_middle, progression_y_middle)
        # edges corners
        self.filter_func_outUnet_array[z_back_out:z_back_in,   x_left_out:x_left_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_left,   progression_y_middle)
        self.filter_func_outUnet_array[z_back_out:z_back_in,   x_right_in:x_right_out, y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_right,  progression_y_middle)
        self.filter_func_outUnet_array[z_back_out:z_back_in,   x_left_in:x_right_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_back,   progression_x_middle, progression_y_down  )
        self.filter_func_outUnet_array[z_back_out:z_back_in,   x_left_in:x_right_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_middle, progression_y_up    )
        self.filter_func_outUnet_array[z_front_in:z_front_out, x_left_out:x_left_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_left,   progression_y_middle)
        self.filter_func_outUnet_array[z_front_in:z_front_out, x_right_in:x_right_out, y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_right,  progression_y_middle)
        self.filter_func_outUnet_array[z_front_in:z_front_out, x_left_in:x_right_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_front,  progression_x_middle, progression_y_down  )
        self.filter_func_outUnet_array[z_front_in:z_front_out, x_left_in:x_right_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_middle, progression_y_up    )
        self.filter_func_outUnet_array[z_back_in:z_front_in,   x_left_out:x_left_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_middle, progression_x_left,   progression_y_down  )
        self.filter_func_outUnet_array[z_back_in:z_front_in,   x_right_in:x_right_out, y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_middle, progression_x_right,  progression_y_down  )
        self.filter_func_outUnet_array[z_back_in:z_front_in,   x_left_out:x_left_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_left,   progression_y_up    )
        self.filter_func_outUnet_array[z_back_in:z_front_in,   x_right_in:x_right_out, y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_right,  progression_y_up    )
        # corners
        self.filter_func_outUnet_array[z_back_out:z_back_in,   x_left_out:x_left_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_back,   progression_x_left,   progression_y_down  )
        self.filter_func_outUnet_array[z_back_out:z_back_in,   x_right_in:x_right_out, y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_back,   progression_x_right,  progression_y_down  )
        self.filter_func_outUnet_array[z_back_out:z_back_in,   x_left_out:x_left_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_left,   progression_y_up    )
        self.filter_func_outUnet_array[z_back_out:z_back_in,   x_right_in:x_right_out, y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_right,  progression_y_up    )
        self.filter_func_outUnet_array[z_front_in:z_front_out, x_left_out:x_left_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_front,  progression_x_left,   progression_y_down  )
        self.filter_func_outUnet_array[z_front_in:z_front_out, x_right_in:x_right_out, y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_front,  progression_x_right,  progression_y_down  )
        self.filter_func_outUnet_array[z_front_in:z_front_out, x_left_out:x_left_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_left,   progression_y_up    )
        self.filter_func_outUnet_array[z_front_in:z_front_out, x_right_in:x_right_out, y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_right,  progression_y_up    )

    def fill_flat_exterior_boundbox(self, boundbox_outer):
        # assign probability 'zero' outside box
        (z_back, z_front, x_left, x_right, y_down, y_up) = boundbox_outer

        self.filter_func_outUnet_array[0:z_back, :,        :       ] = 0.0
        self.filter_func_outUnet_array[z_front:, :,        :       ] = 0.0
        self.filter_func_outUnet_array[:,        0:x_left, :       ] = 0.0
        self.filter_func_outUnet_array[:,        x_right:, :       ] = 0.0
        self.filter_func_outUnet_array[:,        :,        0:y_down] = 0.0
        self.filter_func_outUnet_array[:,        :,        y_up:   ] = 0.0