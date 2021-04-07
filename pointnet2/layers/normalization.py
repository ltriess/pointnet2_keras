#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Larissa Triess"
__email__ = "mail@triess.eu"


from typing import List

import tensorflow as tf


class NeighborhoodNormalization(tf.keras.layers.Layer):
    def __init__(self):
        """ Normalize neighborhoods towards their search points """
        super().__init__()

    def call(self, inputs: List[tf.Tensor], **_) -> tf.Tensor:
        points = inputs[0]  # tf.Tensor(shape=(*, 3), dtype=tf.float32)
        neighborhoods = inputs[1]  # tf.Tensor(shape=(*, K, 3), dtype=tf.float32)

        # Get the unit vector from the new origin for the transformed x axis.
        # u = v / |v| with v being the search point -> new transformed origin.
        with tf.name_scope("unit_vector"):
            unit_vec = tf.div_no_nan(
                points, tf.linalg.norm(points, ord=2, axis=-1, keepdims=True)
            )

        # The transformation matrix T (from local into global) is defined as
        # T = [
        #       [R_00, R_01, R_02, t_0],
        #       [R_10, R_11, R_12, t_1],
        #       [R_20, R_21, R_22, t_2],
        #       [   0,    0,    0,   1]
        #     ]
        # with R being the rotation matrix and t being the translation vector.
        # To transform all coordinates from global to local we use p' = T^(-1) * p
        # with p being homogeneous coordinates (see below).
        # The axes are only rotated in the xy-plane. The z-axis is not converted.
        # The translation vector is equal to the global vector to the search point,
        # since the local coordinate system has its origin in this point.
        # Therefore exactly N points are located in the local neighborhoods.
        # The local x axis is equal to the unit vector of the global search point.
        with tf.name_scope("transformation_matrix"):
            ones = tf.ones_like(unit_vec[..., 0])
            transformation_matrix = tf.convert_to_tensor(
                [
                    [unit_vec[..., 0], -unit_vec[..., 1], 0 * ones, points[..., 0]],
                    [unit_vec[..., 1], unit_vec[..., 0], 0 * ones, points[..., 1]],
                    [0 * ones, 0 * ones, 1 * ones, points[..., 2]],
                    [0 * ones, 0 * ones, 0 * ones, 1 * ones],
                ]
            )  # --> [4, 4, *]
            transformation_matrix = tf.broadcast_to(
                transformation_matrix[..., tf.newaxis],
                shape=tf.concat([[4, 4], tf.shape(neighborhoods)[:-1]], 0),
            )  # --> [4, 4, *, K]
            transformation_matrix = tf.transpose(
                transformation_matrix,
                perm=tf.concat(
                    [tf.range(2, tf.rank(neighborhoods) + 1), [0, 1]], axis=0
                ),
            )  # --> [*, K, 4, 4]

        with tf.name_scope("homogeneous_transformation"):
            # Get homogeneous coordinates for p with p_homo = (p_x, p_y, p_z, 1)
            homo_knn_points = tf.concat(
                [neighborhoods, tf.ones_like(neighborhoods[..., 0:1])], axis=-1
            )  # [*, K, 3] + [*, K, 1] --> [*, K, 4]
            homo_knn_points = homo_knn_points[..., tf.newaxis]  # --> [*, K, 4, 1]
            # Transform neighborhood points into their own coordinate system with
            # p_homo' = T^(-1) * p_homo
            homo_knn_points = tf.matmul(
                tf.linalg.inv(transformation_matrix), homo_knn_points
            )
            # Retrieve points from homogeneous form.
            neighborhoods = homo_knn_points[..., :-1, 0]  # [*, K, 4, 1] --> [*, K, 3]

        return neighborhoods
