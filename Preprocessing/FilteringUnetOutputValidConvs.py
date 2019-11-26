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
from Common.ErrorMessages import *
from OperationImages.BoundingBoxes import *
from Preprocessing.BaseImageGenerator import *



class FilteringUnetOutputValidConvs(BaseImageGenerator):

    type_progression_outside_output_nnet = 'linear'
    avail_type_progression_outside_output_nnet = ['linear', 'quadratic', 'cubic', 'exponential', 'all_outputs_Unet']

    def __init__(self, size_image, size_output_image, is_multiple_outputs_nnet=False):
        self.size_image   = size_image
        self.size_output_image = size_output_image
        self.is_multiple_outputs_nnet = is_multiple_outputs_nnet

        if is_multiple_outputs_nnet:
            self.type_progression = 'all_outputs_Unet'
            if type(size_output_image) != type([]):
                message = 'Wrong input argument \'size_output_image\': \'%s\'. It must be a list of sizes (tuples)...'
                CatchErrorException(message)
        else:
            self.type_progression = 'quadratic'

        self.compute_probabilitymap_output_nnet()
        super(FilteringUnetOutputValidConvs, self).__init__(size_image)


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


    def fill_flat_interior_bounding_box(self, inner_boundbox):
        return NotImplemented

    def fill_progression_between_two_bounding_boxes(self, inner_boundbox, outer_boundbox, prop_val_in=1.0, prop_val_out=0.0):
        return NotImplemented

    def fill_flat_exterior_bounding_box(self, outer_boundbox):
        return NotImplemented


    def update_image_data(self, in_array_shape):
        self.num_images = in_array_shape[0]

    def get_probmap_output_nnet_array(self):
        return self.probmap_output_nnet_array

    def check_correct_dims_image_to_filter(self, in_array_shape):
        if self.is_image_array_without_channels(in_array_shape):
            in_array_size_image = in_array_shape
        else:
            in_array_size_image = in_array_shape[:-2]
        if in_array_size_image == self.size_image:
            return True
        else:
            return False


    def compute_probabilitymap_output_nnet(self):
        self.probmap_output_nnet_array = np.zeros(self.size_image, dtype=FORMATPROBABILITYDATA)

        if self.is_multiple_outputs_nnet:
            # set flat probability equal to 'one' inside inner output of nnet
            # set piecewise probability distribution in between bounding boxes corresponding to outputs of Unet up to 'max_depth'
            # in between bounding boxes, assume quadratic distribution in between two values with diff: 1/num_output_nnet

            num_outputs_nnet = len(size_output_image)
            sizes_bounding_boxes_output = self.size_output_image + [self.size_image]

            inner_bounding_box_output = BoundingBoxes.compute_bounding_box_centered_image_fit_image(self.size_image, sizes_bounding_boxes_output[0])

            self.fill_flat_interior_bounding_box(inner_bounding_box_output)

            for i in range(num_outputs_nnet):
                prop_val_in  = 1.0 - i / float(num_outputs_nnet)
                prop_val_out = 1.0 - (i + 1) / float(num_outputs_nnet)

                inner_bounding_box_output = BoundingBoxes.compute_bounding_box_centered_image_fit_image(self.size_image, sizes_bounding_boxes_output[i])
                outer_bounding_box_output = BoundingBoxes.compute_bounding_box_centered_image_fit_image(self.size_image, sizes_bounding_boxes_output[i+1])

                self.fill_progression_between_two_bounding_boxes(inner_bounding_box_output, outer_bounding_box_output, prop_val_in, prop_val_out)
            # endfor

        else:
            # set flat probability equal to 'one' inside output of nnet
            # set probability distribution (linear, quadratic, ...) in between output of nnet and borders of image

            inner_bounding_box_output = BoundingBoxes.compute_bounding_box_centered_image_fit_image(self.size_image, self.size_output_image)
            image_bounding_box_default = BoundingBoxes.get_default_bounding_box_image(self.size_image)

            self.fill_flat_interior_bounding_box(inner_bounding_box_output)
            self.fill_progression_between_two_bounding_boxes(inner_bounding_box_output, image_bounding_box_default)


    def get_filtered_proboutnnet_array(self, in_array):
        if self.check_correct_dims_image_to_filter(in_array.shape):
            if is_image_array_without_channels(self.size_image, in_array.shape):
                return np.multiply(self.probmap_output_nnet_array, in_array)
            else:
                return np.multiply(self.probmap_output_nnet_array, in_array[..., :])
        else:
            return None


    def get_image(self, in_array):
        return self.get_filtered_proboutnnet_array(in_array)



class FilteringUnetOutputValidConvs2D(FilteringUnetOutputValidConvs):

    def __init__(self, size_image, size_output_image):
        super(FilteringUnetOutputValidConvs2D, self).__init__(size_image, size_output_image)


    def fill_flat_interior_bounding_box(self, inner_boundbox):
        # assign probability 'one' inside box
        (x_left, x_right, y_down, y_up) = inner_boundbox
        self.probmap_output_nnet_array[x_left:x_right, y_down:y_up] = 1.0


    def fill_progression_between_two_bounding_boxes(self, inner_boundbox, outer_boundbox, prop_val_in=1.0, prop_val_out=0.0):
        # assign probability distribution between 'inner' and 'outer' boundboxes, between values 'prop_output_val_in' and 'prop_output_val_out'
        (x_left_in, x_right_in, y_down_in, y_up_in ) = inner_boundbox
        (x_left_out, x_right_out, y_down_out, y_up_out) = outer_boundbox

        progression_x_left  = self.compute_progression_increasing(x_left_out, x_left_in  ) * (prop_val_in - prop_val_out) + prop_val_out
        progression_x_right = self.compute_progression_decreasing(x_right_in, x_right_out) * (prop_val_in - prop_val_out) + prop_val_out
        progression_y_down  = self.compute_progression_increasing(y_down_out, y_down_in  ) * (prop_val_in - prop_val_out) + prop_val_out
        progression_y_up    = self.compute_progression_decreasing(y_up_in,    y_up_out   ) * (prop_val_in - prop_val_out) + prop_val_out
        progression_x_middle = np.ones([x_right_in - x_left_in]) * prop_val_in
        progression_y_middle = np.ones([y_up_in    - y_down_in]) * prop_val_in

        # laterals
        self.probmap_output_nnet_array[x_left_out:x_left_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_2D(progression_x_left,   progression_y_middle)
        self.probmap_output_nnet_array[x_right_in:x_right_out, y_down_in:y_up_in   ] = self.compute_tensor_product_2D(progression_x_right,  progression_y_middle)
        self.probmap_output_nnet_array[x_left_in:x_right_in,   y_down_out:y_down_in] = self.compute_tensor_product_2D(progression_x_middle, progression_y_down  )
        self.probmap_output_nnet_array[x_left_in:x_right_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_2D(progression_x_middle, progression_y_up    )
        # corners
        self.probmap_output_nnet_array[x_left_out:x_left_in,   y_down_out:y_down_in] = self.compute_tensor_product_2D(progression_x_left,   progression_y_down  )
        self.probmap_output_nnet_array[x_right_in:x_right_out, y_down_out:y_down_in] = self.compute_tensor_product_2D(progression_x_right,  progression_y_down  )
        self.probmap_output_nnet_array[x_left_out:x_left_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_2D(progression_x_left,   progression_y_up    )
        self.probmap_output_nnet_array[x_right_in:x_right_out, y_up_in:y_up_out    ] = self.compute_tensor_product_2D(progression_x_right,  progression_y_up    )


    def fill_flat_exterior_bounding_box(self, outer_boundbox):
        # assign probability 'zero' outside box
        (x_left, x_right, y_down, y_up) = outer_boundbox

        self.probmap_output_nnet_array[0:x_left, :] = 0.0
        self.probmap_output_nnet_array[x_right:, :] = 0.0
        self.probmap_output_nnet_array[:, 0:y_down] = 0.0
        self.probmap_output_nnet_array[:, y_up:   ] = 0.0



class FilteringUnetOutputValidConvs3D(FilteringUnetOutputValidConvs):

    def __init__(self, size_image, size_output_image):
        super(FilteringUnetOutputValidConvs3D, self).__init__(size_image, size_output_image)


    def fill_flat_interior_bounding_box(self, inner_boundbox):
        # assign probability 'one' inside box
        (z_back, z_front, x_left, x_right, y_down, y_up) = inner_boundbox
        self.probmap_output_nnet_array[z_back:z_front, x_left:x_right, y_down:y_up] = 1.0


    def fill_progression_between_two_bounding_boxes(self, inner_boundbox, outer_boundbox, prop_val_in=1.0, prop_val_out=0.0):
        # assign probability distribution between 'inner' and 'outer' boundboxes, between values 'prop_output_val_in' and 'prop_output_val_out'
        (z_back_in, z_front_in, x_left_in, x_right_in, y_down_in, y_up_in ) = inner_boundbox
        (z_back_out, z_front_out, x_left_out, x_right_out, y_down_out, y_up_out) = outer_boundbox

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
        self.probmap_output_nnet_array[z_back_in:z_front_in,   x_left_out:x_left_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_left,   progression_y_middle)
        self.probmap_output_nnet_array[z_back_in:z_front_in,   x_right_in:x_right_out, y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_right,  progression_y_middle)
        self.probmap_output_nnet_array[z_back_in:z_front_in,   x_left_in:x_right_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_middle, progression_x_middle, progression_y_down  )
        self.probmap_output_nnet_array[z_back_in:z_front_in,   x_left_in:x_right_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_middle, progression_y_up    )
        self.probmap_output_nnet_array[z_back_out:z_back_in,   x_left_in:x_right_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_middle, progression_y_middle)
        self.probmap_output_nnet_array[z_front_in:z_front_out, x_left_in:x_right_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_middle, progression_y_middle)
        # edges corners
        self.probmap_output_nnet_array[z_back_out:z_back_in,   x_left_out:x_left_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_left,   progression_y_middle)
        self.probmap_output_nnet_array[z_back_out:z_back_in,   x_right_in:x_right_out, y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_right,  progression_y_middle)
        self.probmap_output_nnet_array[z_back_out:z_back_in,   x_left_in:x_right_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_back,   progression_x_middle, progression_y_down  )
        self.probmap_output_nnet_array[z_back_out:z_back_in,   x_left_in:x_right_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_middle, progression_y_up    )
        self.probmap_output_nnet_array[z_front_in:z_front_out, x_left_out:x_left_in,   y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_left,   progression_y_middle)
        self.probmap_output_nnet_array[z_front_in:z_front_out, x_right_in:x_right_out, y_down_in:y_up_in   ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_right,  progression_y_middle)
        self.probmap_output_nnet_array[z_front_in:z_front_out, x_left_in:x_right_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_front,  progression_x_middle, progression_y_down  )
        self.probmap_output_nnet_array[z_front_in:z_front_out, x_left_in:x_right_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_middle, progression_y_up    )
        self.probmap_output_nnet_array[z_back_in:z_front_in,   x_left_out:x_left_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_middle, progression_x_left,   progression_y_down  )
        self.probmap_output_nnet_array[z_back_in:z_front_in,   x_right_in:x_right_out, y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_middle, progression_x_right,  progression_y_down  )
        self.probmap_output_nnet_array[z_back_in:z_front_in,   x_left_out:x_left_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_left,   progression_y_up    )
        self.probmap_output_nnet_array[z_back_in:z_front_in,   x_right_in:x_right_out, y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_middle, progression_x_right,  progression_y_up    )
        # corners
        self.probmap_output_nnet_array[z_back_out:z_back_in,   x_left_out:x_left_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_back,   progression_x_left,   progression_y_down  )
        self.probmap_output_nnet_array[z_back_out:z_back_in,   x_right_in:x_right_out, y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_back,   progression_x_right,  progression_y_down  )
        self.probmap_output_nnet_array[z_back_out:z_back_in,   x_left_out:x_left_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_left,   progression_y_up    )
        self.probmap_output_nnet_array[z_back_out:z_back_in,   x_right_in:x_right_out, y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_back,   progression_x_right,  progression_y_up    )
        self.probmap_output_nnet_array[z_front_in:z_front_out, x_left_out:x_left_in,   y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_front,  progression_x_left,   progression_y_down  )
        self.probmap_output_nnet_array[z_front_in:z_front_out, x_right_in:x_right_out, y_down_out:y_down_in] = self.compute_tensor_product_3D(progression_z_front,  progression_x_right,  progression_y_down  )
        self.probmap_output_nnet_array[z_front_in:z_front_out, x_left_out:x_left_in,   y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_left,   progression_y_up    )
        self.probmap_output_nnet_array[z_front_in:z_front_out, x_right_in:x_right_out, y_up_in:y_up_out    ] = self.compute_tensor_product_3D(progression_z_front,  progression_x_right,  progression_y_up    )


    def fill_flat_exterior_bounding_box(self, outer_boundbox):
        # assign probability 'zero' outside box
        (z_back, z_front, x_left, x_right, y_down, y_up) = outer_boundbox

        self.probmap_output_nnet_array[0:z_back, :, :] = 0.0
        self.probmap_output_nnet_array[z_front:, :, :] = 0.0
        self.probmap_output_nnet_array[:, 0:x_left, :] = 0.0
        self.probmap_output_nnet_array[:, x_right:, :] = 0.0
        self.probmap_output_nnet_array[:, :, 0:y_down] = 0.0
        self.probmap_output_nnet_array[:, :, y_up:   ] = 0.0