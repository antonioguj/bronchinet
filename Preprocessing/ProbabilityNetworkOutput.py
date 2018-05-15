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
from Networks.Networks_NEW import *
from Preprocessing.SlidingWindowImages import *


class ProbabilityNetworkOutput(object):

    type_progression_outside_output_nnet = 'quadratic'
    avail_type_progression_outside_output_nnet = ['linear', 'quadratic', 'cubic', 'exponential', 'all_outputs_Unet']


    def __init__(self, size_image, size_total_image, num_images_total, size_outnnet):
        self.size_image       = size_image
        self.size_total_image = size_total_image
        self.num_images_total = num_images_total
        self.size_outnnet     = size_outnnet
        try:
            # if len(self.size_outnnet) == 1:
            # a bit sloppy: integer doesn't have attribute 'len'
            dummy = len(self.size_outnnet[0])
            self.size_boundbox_outnnet = self.size_outnnet + [self.size_image]
            self.type_progression = 'all_outputs_Unet'
        except TypeError:
            self.type_progression = self.type_progression_outside_output_nnet
            self.size_boundbox_outnnet = [self.size_outnnet] + [self.size_image]


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
    def get_limits_cropImage(size_image, size_outnnet):
        if (size_image == size_outnnet):
            list_out_aux = [[0] + [s_i] for s_i in size_image]
        else:
            list_out_aux = [[(s_i - s_o) / 2] + [(s_i + s_o) / 2] for (s_i, s_o) in zip(size_image, size_outnnet)]
        # flatten out list of lists and return tuple
        return tuple(reduce(lambda el1, el2: el1 + el2, list_out_aux))


    def set_calc_global_prob_outnnet_array(self, input_array):
        self.calc_global_prob_outnnet_array = input_array

    def set_calc_sample_prob_outnnet_array(self, input_array):
        self.calc_sample_prob_outnnet_array = input_array

    def adding_sample_to_global_prob_outnnet_array(self, index, sample_array):
        pass

    def fill_flat_interior_boundbox(self, boundbox_inner):
        pass

    def fill_progression_between_two_boundboxes(self, boundbox_in, boundbox_out, prop_val_in=1.0, prop_val_out=0.0):
        pass

    def fill_flat_exterior_boundbox(self, boundbox_outer):
        pass


    def compute_sample_prob_outnnet_array(self):

        sample_prob_outnnet_array = np.zeros(self.size_image, dtype=FORMATPROPDATA)

        self.set_calc_sample_prob_outnnet_array(sample_prob_outnnet_array)

        if self.type_progression == 'all_outputs_Unet':
            # set flat probability equal to 'one' inside inner output of nnet
            # set piecewise probability distribution in between bounding boxes corresponding to outputs of Unet up to 'max_depth'
            # in between bounding boxes, assume quadratic distribution in between two values with diff: 1/num_output_nnet

            num_output_nnet = len(self.size_outnnet)

            boundbox_inner_outnnet = self.get_limits_cropImage(self.size_image, self.size_boundbox_outnnet[0])

            self.fill_flat_interior_boundbox(boundbox_inner_outnnet)

            for i in range(num_output_nnet):
                prop_val_in  = 1.0 - i / float(num_output_nnet)
                prop_val_out = 1.0 - (i + 1) / float(num_output_nnet)

                boundbox_inner_outnnet = self.get_limits_cropImage(self.size_image, self.size_boundbox_outnnet[i])
                boundbox_outer_outnnet = self.get_limits_cropImage(self.size_image, self.size_boundbox_outnnet[i+1])

                self.fill_progression_between_two_boundboxes(boundbox_inner_outnnet, boundbox_outer_outnnet, prop_val_in, prop_val_out)
            # endfor
        else:
            # set flat probability equal to 'one' inside output of nnet
            # set probability distribution (linear, quadratic, ...) in between output of nnet and borders of image

            boundbox_inner_outnnet = self.get_limits_cropImage(self.size_image, self.size_boundbox_outnnet[0])
            boundbox_size_image    = self.get_limits_cropImage(self.size_image, self.size_boundbox_outnnet[1])

            self.fill_flat_interior_boundbox(boundbox_inner_outnnet)

            self.fill_progression_between_two_boundboxes(boundbox_inner_outnnet, boundbox_size_image)

        return sample_prob_outnnet_array


    def compute(self):

        global_prob_outnnet_array = np.zeros(self.size_total_image, dtype=FORMATPROPDATA)

        self.set_calc_global_prob_outnnet_array(global_prob_outnnet_array)

        sample_prob_outnnet_array = self.compute_sample_prob_outnnet_array()

        for index in range(self.num_images_total):
            self.adding_sample_to_global_prob_outnnet_array(index, sample_prob_outnnet_array)
        # endfor

        # normalize the results by the max. value to account for overlapping images batches
        factor_normalization = np.max(global_prob_outnnet_array)

        return np.divide(global_prob_outnnet_array, factor_normalization)


class ProbabilityNetworkOutput2D(ProbabilityNetworkOutput):

    def __init__(self, size_image, size_total_image, prop_overlap, size_outnnet):

        self.slidingWindow_generator = SlidingWindowImages2D(size_image, prop_overlap, size_total=size_total_image)

        num_images_total = self.slidingWindow_generator.get_num_images()

        super(ProbabilityNetworkOutput2D, self).__init__(size_image, size_total_image, num_images_total, size_outnnet)


    def fill_flat_interior_boundbox(self, boundbox_inner):
        # assign probability 'one' inside box
        (x_left, x_right, y_down, y_up) = boundbox_inner

        self.calc_sample_prob_outnnet_array[x_left:x_right, y_down:y_up] = 1.0

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
        self.calc_sample_prob_outnnet_array[x_left_out:x_left_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_2D(progression_x_left,   progression_y_middle)
        self.calc_sample_prob_outnnet_array[x_right_in:x_right_out, y_down_in:y_up_in   ] = self.compute_tensor_product_2D(progression_x_right,  progression_y_middle)
        self.calc_sample_prob_outnnet_array[x_left_in:x_right_in,   y_down_out:y_down_in] = self.compute_tensor_product_2D(progression_x_middle, progression_y_down  )
        self.calc_sample_prob_outnnet_array[x_left_in:x_right_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_2D(progression_x_middle, progression_y_up    )
        # corners
        self.calc_sample_prob_outnnet_array[x_left_out:x_left_in,   y_down_out:y_down_in] = self.compute_tensor_product_2D(progression_x_left,   progression_y_down  )
        self.calc_sample_prob_outnnet_array[x_right_in:x_right_out, y_down_out:y_down_in] = self.compute_tensor_product_2D(progression_x_right,  progression_y_down  )
        self.calc_sample_prob_outnnet_array[x_left_out:x_left_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_2D(progression_x_left,   progression_y_up    )
        self.calc_sample_prob_outnnet_array[x_right_in:x_right_out, y_up_in:y_up_out    ] = self.compute_tensor_product_2D(progression_x_right,  progression_y_up    )

    def fill_flat_exterior_boundbox(self, boundbox_outer):
        # assign probability 'zero' outside box
        (x_left, x_right, y_down, y_up) = boundbox_outer

        self.calc_sample_prob_outnnet_array[0:x_left, :       ] = 0.0
        self.calc_sample_prob_outnnet_array[x_right:, :       ] = 0.0
        self.calc_sample_prob_outnnet_array[:,        0:y_down] = 0.0
        self.calc_sample_prob_outnnet_array[:,        y_up:   ] = 0.0


    def adding_sample_to_global_prob_outnnet_array(self, index, sample_prob_outnnet_array):

        (x_left, x_right, y_down, y_up) = self.slidingWindow_generator.get_limits_image(index)

        # full_array[x_left:x_right, y_down:y_up, ...] += batch_array
        self.calc_global_prob_outnnet_array[x_left:x_right, y_down:y_up] += sample_prob_outnnet_array


class ProbabilityNetworkOutput3D(ProbabilityNetworkOutput):

    def __init__(self,  size_image, size_total_image, prop_overlap, size_outnnet):

        self.slidingWindow_generator = SlidingWindowImages3D(size_image, prop_overlap, size_total=size_total_image)

        num_images_total = self.slidingWindow_generator.get_num_images()

        super(ProbabilityNetworkOutput3D, self).__init__(size_image, size_total_image, num_images_total, size_outnnet)


    def fill_flat_interior_boundbox(self, boundbox_inner):
        # assign probability 'one' inside box
        (z_back, z_front, x_left, x_right, y_down, y_up) = boundbox_inner

        self.calc_sample_prob_outnnet_array[z_back:z_front, x_left:x_right, y_down:y_up] = 1.0

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
        self.calc_sample_prob_outnnet_array[z_back_in:z_front_in,   x_left_out:x_left_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_left,   progression_y_middle)
        self.calc_sample_prob_outnnet_array[z_back_in:z_front_in,   x_right_in:x_right_out, y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_right,  progression_y_middle)
        self.calc_sample_prob_outnnet_array[z_back_in:z_front_in,   x_left_in:x_right_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_middle, progression_x_middle, progression_y_down  )
        self.calc_sample_prob_outnnet_array[z_back_in:z_front_in,   x_left_in:x_right_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_middle, progression_y_up    )
        self.calc_sample_prob_outnnet_array[z_back_out:z_back_in,   x_left_in:x_right_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_middle, progression_y_middle)
        self.calc_sample_prob_outnnet_array[z_front_in:z_front_out, x_left_in:x_right_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_middle, progression_y_middle)
        # edges corners
        self.calc_sample_prob_outnnet_array[z_back_out:z_back_in,   x_left_out:x_left_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_left,   progression_y_middle)
        self.calc_sample_prob_outnnet_array[z_back_out:z_back_in,   x_right_in:x_right_out, y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_right,  progression_y_middle)
        self.calc_sample_prob_outnnet_array[z_back_out:z_back_in,   x_left_in:x_right_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_back,   progression_x_middle, progression_y_down  )
        self.calc_sample_prob_outnnet_array[z_back_out:z_back_in,   x_left_in:x_right_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_middle, progression_y_up    )
        self.calc_sample_prob_outnnet_array[z_front_in:z_front_out, x_left_out:x_left_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_left,   progression_y_middle)
        self.calc_sample_prob_outnnet_array[z_front_in:z_front_out, x_right_in:x_right_out, y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_right,  progression_y_middle)
        self.calc_sample_prob_outnnet_array[z_front_in:z_front_out, x_left_in:x_right_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_front,  progression_x_middle, progression_y_down  )
        self.calc_sample_prob_outnnet_array[z_front_in:z_front_out, x_left_in:x_right_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_middle, progression_y_up    )
        self.calc_sample_prob_outnnet_array[z_back_in:z_front_in,   x_left_out:x_left_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_middle, progression_x_left,   progression_y_down  )
        self.calc_sample_prob_outnnet_array[z_back_in:z_front_in,   x_right_in:x_right_out, y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_middle, progression_x_right,  progression_y_down  )
        self.calc_sample_prob_outnnet_array[z_back_in:z_front_in,   x_left_out:x_left_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_left,   progression_y_up    )
        self.calc_sample_prob_outnnet_array[z_back_in:z_front_in,   x_right_in:x_right_out, y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_right,  progression_y_up    )
        # corners
        self.calc_sample_prob_outnnet_array[z_back_out:z_back_in,   x_left_out:x_left_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_back,   progression_x_left,   progression_y_down  )
        self.calc_sample_prob_outnnet_array[z_back_out:z_back_in,   x_right_in:x_right_out, y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_back,   progression_x_right,  progression_y_down  )
        self.calc_sample_prob_outnnet_array[z_back_out:z_back_in,   x_left_out:x_left_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_left,   progression_y_up    )
        self.calc_sample_prob_outnnet_array[z_back_out:z_back_in,   x_right_in:x_right_out, y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_right,  progression_y_up    )
        self.calc_sample_prob_outnnet_array[z_front_in:z_front_out, x_left_out:x_left_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_front,  progression_x_left,   progression_y_down  )
        self.calc_sample_prob_outnnet_array[z_front_in:z_front_out, x_right_in:x_right_out, y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_front,  progression_x_right,  progression_y_down  )
        self.calc_sample_prob_outnnet_array[z_front_in:z_front_out, x_left_out:x_left_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_left,   progression_y_up    )
        self.calc_sample_prob_outnnet_array[z_front_in:z_front_out, x_right_in:x_right_out, y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_right,  progression_y_up    )

    def fill_flat_exterior_boundbox(self, boundbox_outer):
        # assign probability 'zero' outside box
        (z_back, z_front, x_left, x_right, y_down, y_up) = boundbox_outer

        self.calc_sample_prob_outnnet_array[0:z_back, :,        :       ] = 0.0
        self.calc_sample_prob_outnnet_array[z_front:, :,        :       ] = 0.0
        self.calc_sample_prob_outnnet_array[:,        0:x_left, :       ] = 0.0
        self.calc_sample_prob_outnnet_array[:,        x_right:, :       ] = 0.0
        self.calc_sample_prob_outnnet_array[:,        :,        0:y_down] = 0.0
        self.calc_sample_prob_outnnet_array[:,        :,        y_up:   ] = 0.0


    def adding_sample_to_global_prob_outnnet_array(self, index, sample_prob_outnnet_array):

        (z_back, z_front, x_left, x_right, y_down, y_up) = self.slidingWindow_generator.get_limits_image(index)

        # full_array[z_back:z_front, x_left:x_right, y_down:y_up, ...] += batch_array
        self.calc_global_prob_outnnet_array[z_back:z_front, x_left:x_right, y_down:y_up] += sample_prob_outnnet_array