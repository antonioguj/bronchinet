
from typing import Tuple
import numpy as np
import argparse

from dataloaders.imagefilereader import ImageFileReader

_EPS = 1.0e-10


def get_indexes_canditate_inside_shape(point_center: Tuple[float, float, float],
                                       max_dist_2cen: float
                                       ) -> np.array:
    min_index_x = int(np.floor(point_center[0] - max_dist_2cen))
    max_index_x = int(np.ceil(point_center[0] + max_dist_2cen))
    min_index_y = int(np.floor(point_center[1] - max_dist_2cen))
    max_index_y = int(np.ceil(point_center[1] + max_dist_2cen))
    min_index_z = int(np.floor(point_center[2] - max_dist_2cen))
    max_index_z = int(np.ceil(point_center[2] + max_dist_2cen))
    indexes_x = np.arange(min_index_x, max_index_x + 1)
    indexes_y = np.arange(min_index_y, max_index_y + 1)
    indexes_z = np.arange(min_index_z, max_index_z + 1)
    return np.stack(np.meshgrid(indexes_x, indexes_y, indexes_z, indexing='ij'), axis=3)


def create_sphere_mask(inout_image: np.ndarray,
                       point_center: Tuple[float, float, float] = None,
                       radius: float = 50.0
                       ) -> np.ndarray:
    if not point_center:
        point_center = (inout_image.shape[0] / 2,
                        inout_image.shape[1] / 2,
                        inout_image.shape[2] / 2)

    # candidates: subset of all possible indexes where to check condition for blank shape -> to save time
    indexes_candits_inside = get_indexes_canditate_inside_shape(point_center, radius)

    # relative position of candidate indexes to center
    points_rel2center_candits_inside = indexes_candits_inside - point_center

    # distance to center -> compute the norm of relative position vectors
    dist_rel2center_candits_inside = np.linalg.norm(points_rel2center_candits_inside, axis=3)

    # condition for sphere: distance to center is less than radius
    is_indexes_inside = np.abs(dist_rel2center_candits_inside) <= radius
    # array of ['True', 'False'], with 'True' for indexes that are inside the sphere

    indexes_inside = indexes_candits_inside[is_indexes_inside]

    (indexes_in_x, indexes_in_y, indexes_in_z) = np.transpose(indexes_inside)
    inout_image[indexes_in_z, indexes_in_y, indexes_in_x] = 1

    return inout_image


def create_cylinder_mask(inout_image: np.ndarray,
                         point_center: Tuple[float, float, float] = None,
                         vector_axis: Tuple[float, float, float] = None,
                         radius_base: float = 50.0,
                         length_axis: float = 100.0
                         ) -> np.ndarray:
    if not point_center:
        point_center = (inout_image.shape[0] / 2,
                        inout_image.shape[1] / 2,
                        inout_image.shape[2] / 2)

    if not vector_axis:
        vector_axis = (0.0, 0.0, 1.0)

    norm_vector_axis = np.sqrt(np.dot(vector_axis, vector_axis))
    unit_vector_axis = np.array(vector_axis) / norm_vector_axis
    half_length_axis = length_axis / 2.0

    # candidates: subset of all possible indexes where to check condition for blank shape -> to save time
    dist_corner_2center = np.sqrt(radius_base ** 2 + half_length_axis ** 2)
    indexes_candits_inside = get_indexes_canditate_inside_shape(point_center, dist_corner_2center)

    # relative position of candidate indexes to center
    points_rel2center_candits_inside = indexes_candits_inside - point_center

    # distance to center, parallel to axis -> dot product of distance vectors with 'vector_axis'
    dist_rel2center_parall_axis_candits = np.dot(points_rel2center_candits_inside, unit_vector_axis)

    # distance to center, perpendicular to axis -> Pythagoras (distance_2center ^2 - distance_2center_parall_axis ^2)
    dist_rel2center_perpen_axis_candits = np.sqrt(np.square(np.linalg.norm(points_rel2center_candits_inside, axis=3))
                                                  - np.square(dist_rel2center_parall_axis_candits) + _EPS)

    # conditions for cylinder: 1) distance to center, parallel to axis, is less than 'half_length_axis'
    #                          2) distance to center, perpendicular to axis, is less than 'radius_base'
    is_indexes_inside_cond1 = np.abs(dist_rel2center_parall_axis_candits) <= half_length_axis
    is_indexes_inside_cond2 = np.abs(dist_rel2center_perpen_axis_candits) <= radius_base

    is_indexes_inside = np.logical_and(is_indexes_inside_cond1, is_indexes_inside_cond2)
    # array of ['True', 'False'], with 'True' for indexes that are inside the cylinder

    indexes_inside = indexes_candits_inside[is_indexes_inside]

    (indexes_in_x, indexes_in_y, indexes_in_z) = np.transpose(indexes_inside)
    inout_image[indexes_in_z, indexes_in_y, indexes_in_x] = 1

    return inout_image


def create_cuboid_mask(inout_image: np.ndarray,
                       point_center: Tuple[float, float, float] = None,
                       vector_axis: Tuple[float, float, float] = None,
                       vector_seg1_base: Tuple[float, float, float] = None,
                       len_seg_base: float = 100.0,
                       length_axis: float = 100.0
                       ) -> np.ndarray:
    if not point_center:
        point_center = (inout_image.shape[0] / 2,
                        inout_image.shape[1] / 2,
                        inout_image.shape[2] / 2)

    if not vector_axis:
        vector_axis = (0.0, 0.0, 1.0)

    if not vector_seg1_base:
        # try perpendicular vectors to 'vector_axis'
        vector_perpen_2axis_1 = (0.0, -vector_axis[2], vector_axis[1])
        vector_perpen_2axis_2 = (-vector_axis[2], 0.0, vector_axis[0])
        vector_perpen_2axis_3 = (-vector_axis[1], vector_axis[0], 0.0)
        vectors_perpen_2axis_try = [vector_perpen_2axis_1, vector_perpen_2axis_2, vector_perpen_2axis_3]

        for vector_perpen_2axis in vectors_perpen_2axis_try:
            # check the perpendicular vector is null
            if np.all(np.abs(vector_perpen_2axis) < _EPS):
                continue
            else:
                vector_seg1_base = vector_perpen_2axis
                break

    # second vector perpendicular to 'vector_axis'
    vector_seg2_base = np.cross(vector_axis, vector_seg1_base)

    norm_vector_axis = np.sqrt(np.dot(vector_axis, vector_axis))
    unit_vector_axis = np.array(vector_axis) / norm_vector_axis
    norm_vector_seg1_base = np.sqrt(np.dot(vector_seg1_base, vector_seg1_base))
    unit_vector_seg1_base = np.array(vector_seg1_base) / norm_vector_seg1_base
    norm_vector_seg2_base = np.sqrt(np.dot(vector_seg2_base, vector_seg2_base))
    unit_vector_seg2_base = np.array(vector_seg2_base) / norm_vector_seg2_base
    half_length_axis = length_axis / 2.0
    half_len_seg_base = len_seg_base / 2.0

    # candidates: subset of all possible indexes where to check condition for blank shape -> to save time
    dist_corner_2center = np.sqrt(2 * half_len_seg_base ** 2 + half_length_axis ** 2)
    indexes_candits_inside = get_indexes_canditate_inside_shape(point_center, dist_corner_2center)

    # relative position of candidate indexes to center
    points_rel2center_candits_inside = indexes_candits_inside - point_center

    # distance to center, parallel to axis -> dot product of distance vectors with 'vector_axis'
    dist_rel2center_parall_axis_candits = np.dot(points_rel2center_candits_inside, unit_vector_axis)

    # candidate distance vectors (component) perpendicular to 'vector_axis'
    points_rel2center_perpen_axis_candits = \
        points_rel2center_candits_inside - dist_rel2center_parall_axis_candits[..., np.newaxis] * unit_vector_axis

    # dist. to center, perpen. to axis, parall. to seg1 -> dot product of perpen. distance vecs. with 'vector_seg1_base'
    dist_rel2cent_perpen_axis_parall_seg1_candits = np.dot(points_rel2center_perpen_axis_candits, unit_vector_seg1_base)

    # dist. to center, perpendicular to axis and seg1 -> dot product of perpen. distance vectors with 'vector_seg2_base'
    dist_rel2cent_perpen_axis_perpen_seg1_candits = np.dot(points_rel2center_perpen_axis_candits, unit_vector_seg2_base)

    # conditions cuboid: 1) dist. to center, parallel to axis: is less than 'half_length_axis'
    #                    2) dist. to center, perpen. to axis and parall. to seg1, is less than 'half_len_seg_base'
    #                    3) dist. to center, perpendicular to axis and seg1, is less than 'half_len_seg_base'
    is_indexes_inside_cond1 = np.abs(dist_rel2center_parall_axis_candits) <= half_length_axis
    is_indexes_inside_cond2 = np.abs(dist_rel2cent_perpen_axis_parall_seg1_candits) <= half_len_seg_base
    is_indexes_inside_cond3 = np.abs(dist_rel2cent_perpen_axis_perpen_seg1_candits) <= half_len_seg_base

    is_indexes_inside = \
        np.logical_and(is_indexes_inside_cond1, np.logical_and(is_indexes_inside_cond2, is_indexes_inside_cond3))
    # array of ['True', 'False'], with 'True' for indexes that are inside the cylinder

    indexes_inside = indexes_candits_inside[is_indexes_inside]

    (indexes_in_x, indexes_in_y, indexes_in_z) = np.transpose(indexes_inside)
    inout_image[indexes_in_z, indexes_in_y, indexes_in_x] = 1

    return inout_image


def main(args):

    # Create image
    out_shape_image = (256, 256, 256)
    out_image = np.zeros(out_shape_image)
    out_image = create_cuboid_mask(out_image,
                                   vector_axis=(1.0, 1.0, 1.0),
                                   len_seg_base=100.0,
                                   length_axis=100.0)

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
