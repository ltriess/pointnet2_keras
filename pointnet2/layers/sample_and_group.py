#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Implementation of PointNet++ Sampling and Grouping

The code is licensed under the LICENSE in the root of this repository.
The original code (https://github.com/charlesq34/pointnet2) is licensed under
the MIT License Copyright (c) 2017 Charles R. Qi and
Copyright (c) 2017, Geometric Computation Group of Stanford University
"""

__author__ = "Larissa Triess"
__email__ = "mail@triess.eu"


import tensorflow as tf
from my_tf_ops.knn_op import k_nearest_neighbor_op as get_knn

from ..layers.normalization import NeighborhoodNormalization
from ..layers.sampling import RandomPointSampling
from ..tf_ops.grouping import query_ball_point
from ..tf_ops.sampling import farthest_point_sample


def group(data: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
    """Group data according to indices.

    Example:
        data (B, N, 3), indices (B, M) --> (B, M, 3)
        data (B, N, C), indices (B, M, K) --> (B, M, K, C)

    Arguments:
        data : tf.Tensor(shape=(*, C), dtype=tf.float32)
            Data of rank R+1, last dimension are the channels to group.
        indices : tf.Tensor(shape=(*,), dtype=tf.int32)
            Indices of rank Q with Q >= R and first R dimensions in same shape as data.

    Returns:
        gathered : tf.Tensor(shape=(*, C), dtype=tf.float32)
            The gathered data of rank Q+1.

    """

    # Remember the original shapes.
    data_shape = data.shape.as_list()
    indices_shape = indices.shape.as_list()
    tf_indices_shape = tf.shape(indices)

    # Flatten the dimensions.
    data = tf.reshape(data, shape=(-1, data_shape[-1]))
    indices = tf.reshape(indices, shape=(-1,))

    # Gather the data.
    out = tf.gather(data, indices)

    # Reshape to original shape with extra dimension and set all known shapes.
    out = tf.reshape(out, shape=tf.concat([tf_indices_shape, data_shape[-1:]], axis=-1))
    out.set_shape((*indices_shape, data_shape[-1]))

    return out


class SampleAndGroup(tf.keras.layers.Layer):
    """ Sampling and Grouping """

    def __init__(
        self,
        num_queries: int,
        num_neighbors: int,
        radius: float = None,
        use_knn: bool = False,
        use_xyz: bool = True,
        sampling: str = "farthest",
        normalization: str = "trans",
    ):
        """Initialize Sampling and Grouping Layer

        Arguments:
            num_queries : int
                The number of farthest points to sample.
            num_neighbors : int
                The number of points to sample in the radius or knn search.
            radius : float
                The radius for the search if use_knn is `False`. (Default is `None`).
            use_knn : bool
                If `True` use KNN search instead of radius search. (Default is `False`).
            use_xyz : bool
                If `True` concat XYZ with local point features, otherwise just
                use point features. (Default is `True`).
            sampling : str, one of `farthest`, `random`
                Either use Farthest or Random Point Sampling.
            normalization : str, one of `trans`, `transrot`
                Either use translation for normalization or translation and rotation.

        Raises:
            ValueError if use_knn is `False` and radius is `None`

        """
        super(SampleAndGroup, self).__init__()
        self.use_xyz = use_xyz

        if not use_knn and radius is None:
            raise ValueError(
                "You want to use radius search, but did not provide a radius."
            )

        # Get the sampling function.
        if sampling == "farthest":
            # Get the M farthest points in points with M = num_queries.
            self.sampling = lambda points: farthest_point_sample(num_queries, points)
        elif sampling == "random":
            # Get M random query points in points with M = num_queries.
            self.sampling = lambda points: RandomPointSampling(num_queries)([points])
        else:
            raise ValueError("Unknown sampling method {m}.".format(m=sampling))

        # Get the neighborhood indices function.
        if use_knn:
            # Get the indices of the K nearest points within points
            # with respect to the query locations.
            self.get_neighbor_indices = lambda points, queries: tf.map_fn(
                lambda x: get_knn(x[0], x[1], num_neighbors)[0],
                [points, queries],
                dtype=tf.int32,
            )
        else:
            # Get the indices of K samples within the search radius
            # with respect to the query locations.
            self.get_neighbor_indices = lambda points, queries: query_ball_point(
                radius, num_neighbors, points, queries
            )[0]

        # Get the normalization function.
        if normalization == "trans":
            self.normalization = (
                lambda neighborhoods, queries: neighborhoods
                - queries[:, :, tf.newaxis, :]
            )
        elif normalization == "transrot":
            norm = NeighborhoodNormalization()
            self.normalization = lambda neighborhoods, queries: norm(
                [queries, neighborhoods]
            )
        else:
            raise ValueError(
                "Unknown normalization method {m}.".format(m=normalization)
            )

    def call(self, points: tf.Tensor, features: tf.Tensor = None):
        """Call of Sampling and Grouping layer

        Arguments:
            points : tf.Tensor(shape=(batch_size, n, 3), dtype=tf.float32)
                The xyz points with n number of points.
            features : tf.Tensor(shape=(batch_size, n, c), dtype=tf.float32)
                The features with n data points and c channels.
                If `None`, will use points as features. (Default is `None`).

        Returns:
            query_points : tf.Tensor(shape=(batch_size, num_queries, 3), dtype=tf.float32)
                The M points sampled from points.
            grouped_features : tf.Tensor(shape=(batch_size, num_queries, num_neighbors, o), dtype=tf.float32)
                The sampled and grouped features. If use_xyz is `True`, then concat
                the local features, then o = 3 + c. If use_xyz is `False`, only use the
                global features, then o = c.
            indices : tf.Tensor(shape=(batch_size, num_queries, num_neighbors), dtype=tf.int32)
                The indices of the local points as in n features.
            grouped_points_normalized : tf.Tensor(shape=(batch_size, num_queries, num_neighbors, 3), dtype=tf.float32)
                The sampled and grouped points, translation normalized towards the
                farthest sampled points in local regions.
            grouped_points : tf.Tensor(shape=(batch_size, num_queries, num_neighbors, 3), dtype=tf.float32)
                The sampled and grouped points, without normalization.

        """

        with tf.name_scope(self.name):

            # Sample M query indices from N points: points (B, N, 3) --> indices (B, N,)
            query_indices = self.sampling(points)
            # Group the M sampled points from selected indices. --> samples (B, M, 3)
            query_points = group(points, query_indices)
            # Get the indices of the points belonging to the neighborhood.
            indices = self.get_neighbor_indices(points, query_points)
            # Gather the points with indices to form neighborhoods.
            # points (B, N, 3) --> grouped points (B, M, K, 3)
            grouped_points = group(points, indices)
            # Normalization of translation for local regions.
            grouped_points_normalized = self.normalization(grouped_points, query_points)

            if features is None:
                # If no features give, use points as features.
                grouped_features = grouped_points_normalized
            else:
                # Group the features according to indices.
                # features (B, N, C) --> grouped features (B, M, K, C)
                grouped_features = group(features, indices)

                if self.use_xyz:
                    # Concat the points with the local features.
                    # grouped_points (B, M, K, 3) + grouped_features (B, M, K, C)
                    # --> (B, M, K, 3 + C)
                    grouped_features = tf.concat(
                        [grouped_points_normalized, grouped_features], axis=-1
                    )

        return (
            query_points,
            grouped_features,
            indices,
            grouped_points_normalized,
            grouped_points,
        )


class SampleAndGroupAll(tf.keras.layers.Layer):
    """Sampling and Grouping

    Note:
        Equivalent to SampleAndGroup(num_points=1, radius=inf)
        using (0, 0, 0) as the centroid.
    """

    def __init__(self, use_xyz: bool = True):
        """Initialize Sampling and Grouping layer

        Arguments:
            use_xyz : bool
                If `True` concat XYZ with local point features, otherwise just
                use point features. (Default is `True`).
        """
        super(SampleAndGroupAll, self).__init__()
        self.use_xyz = use_xyz

    def call(self, points: tf.Tensor, features: tf.Tensor = None):
        """Call of Sampling and Grouping layer

        Arguments:
            points : tf.Tensor(shape=(batch_size, n, 3), dtype=tf.float32)
                The input with n number of points given in xyz.
            features : tf.Tensor(shape=(batch_size, n, c), dtype=tf.float32)
                The features with n number of data points and c channels.
                If `None` will use points as features. (Default is `None`).

        Returns:
            sampled_points : tf.Tensor(shape=(batch_size, 1, 3), dtype=tf.float32)
                The centroid point as (0, 0, 0).
            grouped_features : tf.Tensor(shape=(batch_size, 1, n, o), dtype=tf.float32)
                The sampled and grouped features. If use_xyz is `True`, then concat
                the local features, then o = 3 + c. If use_xyz is `False`, only use the
                global features, then o = c.
            indices : tf.Tensor(shape=(batch_size, 1, n), dtype=tf.int32)
                The indices of the local points as in n features.
            grouped_points : tf.Tensor(shape=(batch_size, 1, n, 3), dtype=tf.float32)
                The sampled and grouped points relative to the sampled_points (0, 0, 0).

        """

        with tf.name_scope(self.name):

            batch_size = tf.shape(points)[0]
            num_points = tf.shape(points)[1]

            # Create sampled_points with shape (B, 1, 3).
            sampled_points = tf.zeros_like(points)[:, 0:1, :]

            # Create the indices as a range directly over points.
            # (B, 1, N) with N being the number of points.
            indices = tf.range(num_points, dtype=tf.int32)[tf.newaxis, tf.newaxis, :]
            indices = tf.broadcast_to(indices, (batch_size, 1, num_points))

            # Fake gather of the points.
            # points (B, N, 3) --> grouped points (B, 1, N, 3)
            grouped_points = points[:, tf.newaxis, :, :]

            if features is None:
                # If no features given, use points as features.
                grouped_features = grouped_points
            else:
                # Fake gather of the features.
                # features (B, N, C) --> grouped features (B, 1, N, C)
                grouped_features = features[:, tf.newaxis, :, :]

                if self.use_xyz:
                    # Concat the points with the local features.
                    # grouped_points (B, 1, N, 3) + grouped_features (B, 1, N, C)
                    # --> (B, 1, N, 3 + C)
                    grouped_features = tf.concat(
                        [grouped_points, grouped_features], axis=-1
                    )

        return sampled_points, grouped_features, indices, grouped_points, grouped_points
