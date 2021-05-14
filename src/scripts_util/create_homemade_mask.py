
from typing import Tuple
import numpy as np
import argparse

from dataloaders.imagefilereader import ImageFileReader


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

    # relative distance of candidate indexes to center
    locs_rel2center_candits = indexes_candits_inside - point_center

    # condition for sphere: if distance to center is less than radius
    is_indexes_inside = np.linalg.norm(locs_rel2center_candits, axis=3) <= radius
    # array of ['True', 'False'], with 'True' for indexes that are inside the sphere

    indexes_inside = indexes_candits_inside[is_indexes_inside]

    (indexes_x, indexes_y, indexes_z) = np.transpose(indexes_inside)
    inout_image[indexes_z, indexes_y, indexes_x] = 1

    return inout_image


def create_cylinder_mask(inout_image: np.ndarray,
                         point_center: Tuple[float, float, float] = None,
                         vector_axis: Tuple[float, float, float] = None,
                         radius_base: float = 50.0,
                         length_axis: float = 50.0
                         ) -> np.ndarray:
    if not point_center:
        point_center = (inout_image.shape[0] / 2,
                        inout_image.shape[1] / 2,
                        inout_image.shape[2] / 2)

    if not vector_axis:
        vector_axis = (0.0, 0.0, 1.0)

    norm_vector_axis = np.sqrt(np.dot(vector_axis, vector_axis))
    unit_vector_axis = vector_axis / norm_vector_axis
    half_length_axis = length_axis / 2.0

    # candidates: subset of all possible indexes where to check condition for blank shape -> to save time
    dist_corner_2cen = np.sqrt(radius_base ** 2 + half_length_axis ** 2)
    indexes_candits_inside = get_indexes_canditate_inside_shape(point_center, dist_corner_2cen)

    # relative distance of candidate indexes to center
    locs_rel2center_candits = indexes_candits_inside - point_center

    # conditions for cylinder: 1) if distance to center, parallel to axis, is less than length_axis
    #                          2) if distance to center, perpendicular to axis, is less than radius_base
    dist_rel2center_parall_axis_candits = np.abs(np.dot(locs_rel2center_candits, unit_vector_axis))

    is_indexes_inside_parall = dist_rel2center_parall_axis_candits <= half_length_axis
    is_indexes_inside_perpen = np.sqrt(np.square(np.linalg.norm(locs_rel2center_candits, axis=3))
                                       - np.square(dist_rel2center_parall_axis_candits)) <= radius_base

    is_indexes_inside = np.logical_and(is_indexes_inside_parall, is_indexes_inside_perpen)
    # array of ['True', 'False'], with 'True' for indexes that are inside the cylinder

    indexes_inside = indexes_candits_inside[is_indexes_inside]

    (indexes_x, indexes_y, indexes_z) = np.transpose(indexes_inside)
    inout_image[indexes_z, indexes_y, indexes_x] = 1

    return inout_image


# def create_cuboid_in_image(inout_image: np.ndarray,
#                            coords0: Tuple[int, int, int] = None,
#                            halfsides: Tuple[int, int, int] = (50, 50, 50)
#                            ) -> np.ndarray:
#     if not coords0:
#         coords0 = (int(inout_image.shape[0] / 2), int(inout_image.shape[1] / 2), int(inout_image.shape[2] / 2))
#
#     (z_min, z_max) = (coords0[0] - halfsides[0], coords0[0] + halfsides[0])
#     (x_min, x_max) = (coords0[1] - halfsides[1], coords0[1] + halfsides[1])
#     (y_min, y_max) = (coords0[2] - halfsides[2], coords0[2] + halfsides[2])
#
#     indexes_z_cuboid = []
#     indexes_x_cuboid = []
#     indexes_y_cuboid = []
#     for iz in range(z_min, z_max, 1):
#         for ix in range(x_min, x_max, 1):
#             for iy in range(y_min, y_max, 1):
#                 indexes_z_cuboid.append(iz)
#                 indexes_x_cuboid.append(ix)
#                 indexes_y_cuboid.append(iy)
#             # endfor
#         # endfor
#     # endfor
#     indexes_cuboid = [indexes_z_cuboid, indexes_y_cuboid, indexes_y_cuboid]
#
#     inout_image[indexes_cuboid] = 1
#     return inout_image


def main(args):

    # Create image
    out_shape_image = (256, 256, 256)
    out_image = np.zeros(out_shape_image)
    out_image = create_cylinder_mask(out_image,
                                     vector_axis=(1.0,1.0,1.0),
                                     radius_base=50.0,
                                     length_axis=50.0)

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
