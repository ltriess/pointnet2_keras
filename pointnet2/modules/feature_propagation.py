#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Larissa Triess"
__email__ = "mail@triess.eu"


from typing import List

import tensorflow as tf
from my_tf_ops.knn_op import k_nearest_neighbor_op as get_knn

from ..layers.sample_and_group import group


class FeaturePropagationModule(tf.keras.models.Model):
    """PointNet Feature Propagation Module

    Arguments:
        mlp_units : List[int]
            Output units for each point-wise mlp.
        feature_norm : str
            The feature normalization to use. Can be `batch` for batch normalization
            or `layer` for layer normalization. If None, no normalization is applied.
        num_neighbors : int
            The number of neighbors to consider for interpolation. Default is 3.

    Raises:
        ValueError if feature normalization is not valid.

    """

    def __init__(
        self, mlp_units: List[int], feature_norm: str = None, num_neighbors: int = 3
    ):
        super().__init__()

        self.K = num_neighbors

        if feature_norm not in {None, "batch", "layer"}:
            raise ValueError(f"Received unknown feature norm `{feature_norm}`!")

        # Find the three nearest neighbors for points_hr in points_lr.
        self.get_neighbor_indices_and_square_distances = (
            lambda points_hr, points_lr: tf.map_fn(
                lambda x: list(get_knn(x[0], x[1], self.K)),
                [points_lr, points_hr],
                dtype=[tf.int32, tf.float32],
                name="feature_propagation_knn",
            )
        )

        self.mlp = tf.keras.models.Sequential(name="feature_propagation_mlp")
        for unit in mlp_units:
            self.mlp.add(
                tf.keras.layers.Conv1D(unit, kernel_size=1, strides=1, padding="valid")
            )
            if feature_norm == "batch":
                self.mlp.add(tf.keras.layers.BatchNormalization())
            elif feature_norm == "layer":
                self.mlp.add(tf.keras.layers.LayerNormalization())
            else:
                pass
            self.mlp.add(tf.keras.layers.LeakyReLU())

    def call(
        self,
        inputs: List[tf.Tensor],
        training: tf.Tensor = None,
        mask: tf.Tensor = None,
    ):
        """Call of PointNet Feature Propagation

        Arguments:
            inputs : List[tf.Tensor] of length 4
                Must contain the following tensors:
                [0] - tf.Tensor(shape=(B, M[i-1], 3), dtype=tf.float32)
                    xyz-points at level i-1
                [1] - tf.Tensor(shape=(B, M[i], 3), dtype=tf.float32)
                    xyz-points at level i
                [2] - tf.Tensor(shape=(B, M[i-1], C[i-1]), dtype=tf.float32)
                    features at level i-1 (can be None)
                [3] - tf.Tensor(shape=(B, M[i], C[i]), dtype=tf.float32)
                    features at level i
                M[x] are the number of points or neighbor hoods at abstraction level x
                with M[x] < M[x-1] (level i is sparser than level i-1).
            training: tf.Tensor(shape=(), dtype=tf.bool)
            mask: tf.Tensor

        Returns:
            tf.Tensor(shape=(B, M[i-1], mlp[-1]), dtype=tf.float32)
        """

        if not isinstance(inputs, list):
            raise ValueError("Inputs must be a list of tensors!")
        if not len(inputs) == 4:
            raise ValueError(
                "Feature propagation module must be called with a list of four tensors. "
                "See documentation!"
            )

        points_hr = inputs[0]  # (B, M[i-1], 3)
        points_lr = inputs[1]  # (B, M[i], 3)
        features_hr = inputs[2]  # (B, M[i-1], C[i-1]) or None
        features_lr = inputs[3]  # (B, M[i], C[i])

        indices, distances = self.get_neighbor_indices_and_square_distances(
            points_hr, points_lr
        )  # 2x(B, M[i-1], K) with K=3
        distances = tf.maximum(distances, 1e-10)  # avoid diving by zero

        # Compute the weighting factor for each neighbor.
        distances_inv = tf.divide(1.0, distances)  # (B, M[i-1], K)
        weight = distances_inv / tf.reduce_sum(distances_inv, axis=-1, keepdims=True)

        check = tf.compat.v1.assert_equal(
            tf.shape(points_lr)[1],
            tf.shape(features_lr)[1],
            message="Number of points and number of features does not match!",
        )
        with tf.control_dependencies([check]):
            # Gather three features from points_lr to match one group in points_hr.
            # (B, M[i], C) and (B, M[i-1], K) --> (B, M[i-1], K, C[i])
            grouped_features = group(features_lr, indices)
        # Interpolate the feature from the K neighbors.
        # Weighted sum over K reduces dimension to (B, M[i-1], C[i]).
        interpolated_features = tf.reduce_sum(
            grouped_features * weight[..., tf.newaxis], axis=2
        )

        if features_hr is not None:
            # Concatenate original and interpolated features to (B, M[i-1], C[i]+C[i-1]).
            interpolated_features = tf.concat(
                [interpolated_features, features_hr], axis=-1
            )

        # Compute new features from interpolations.
        processed_features = self.mlp(interpolated_features)  # (B, M[i-1], mlp[-1])
        return processed_features
