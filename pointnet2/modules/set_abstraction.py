#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Larissa Triess"
__email__ = "mail@triess.eu"


from typing import List

import tensorflow as tf

from ..layers.pooling import AvgPool, MaxAndAvgPool, MaxPool, WeightedAvgPool
from ..layers.sample_and_group import SampleAndGroup, SampleAndGroupAll


class SetAbstractionModule(tf.keras.models.Model):
    """ PointNet Set Abstraction Module"""

    def __init__(
        self,
        mlp_point: List[int],
        mlp_region: List[int] = None,
        num_queries: int = None,
        num_neighbors: int = None,
        radius: float = None,
        group_all: bool = False,
        use_knn: bool = False,
        use_xyz: bool = True,
        pooling: str = "max",
        batch_norm: bool = True,
        sampling: str = "farthest",
        normalization: str = "trans",
        name: str = None,
    ):
        """Initialize Set Abstraction Module

        Arguments:
            mlp_point : List[int]
                A list of integers for the output sizes of the MLP on each point.
            mlp_region : List[int]
                A list of integers for the output sizes of the MLP on each region.
                If `None`, no further processing on the regions is conducted.
            num_queries : int
                The number of points sampled in farthest point sampling.
            num_neighbors : int
                The number of points sampled in each local region.
            radius : float
                The search radius in the local region if use_knn is `False`.
            group_all : bool
                If `True`, then group all points into one point cloud. This is
                equivalent to num_points=1, radius=inf, num_samples=N with N being the
                number of points.
            use_knn : bool
                If `True`, use KNN search. If `False`, use radius search.
            use_xyz : bool
                If `True`, concat points to local point features. If `False`, only use
                local point features.
            sampling : str
                Either use Farthest or Random Point Sampling.
            normalization : str, one of `trans`, `transrot`
                Either use translation for normalization or translation and rotation.
            name : str
                Name of the model.

        Raises:
            ValueError if use_knn is `False` and radius is `None`
            ValueError if group_all is `False` and one of num_points, num_samples, or
                radius is `None`

        """
        super(SetAbstractionModule, self).__init__(name=name)

        if radius is None and use_knn is False and group_all is False:
            raise ValueError(
                "You want to use radius search, but did not provide a radius."
            )
        if not group_all and any(x is None for x in [num_queries, num_neighbors]):
            raise ValueError(
                "num_points and num_samples must be set to a valid value. "
                "I got {0}.".format((num_queries, num_neighbors))
            )

        # Sampling and Grouping.
        if group_all:
            self.sampling_and_grouping = SampleAndGroupAll(use_xyz=use_xyz)
        else:
            self.sampling_and_grouping = SampleAndGroup(
                num_queries=num_queries,
                num_neighbors=num_neighbors,
                radius=radius,
                use_knn=use_knn,
                use_xyz=use_xyz,
                sampling=sampling,
                normalization=normalization,
            )

        # Point Feature Embedding.
        self.point_feature_embedding = tf.keras.Sequential(name="point_feature_embed")
        for out_channels in mlp_point:
            self.point_feature_embedding.add(
                tf.keras.layers.Conv2D(
                    filters=out_channels, kernel_size=(1, 1), strides=(1, 1)
                )
            )
            if batch_norm:
                self.point_feature_embedding.add(
                    tf.keras.layers.BatchNormalization(axis=3, momentum=0.9)
                )
            self.point_feature_embedding.add(tf.keras.layers.LeakyReLU())

        # Pooling in Local Regions.
        if pooling == "max":
            self.pool_local_regions = MaxPool()
        elif pooling == "avg":
            self.pool_local_regions = AvgPool()
        elif pooling == "weighted_avg":
            self.pool_local_regions = WeightedAvgPool()
        elif pooling == "max_and_avg":
            self.pool_local_regions = MaxAndAvgPool()

        # Region Feature Embedding.
        self.region_feature_embedding = tf.keras.Sequential(name="region_feature_embed")
        if mlp_region is None:
            # Just a dummy identity layer for more convenience in the call function.
            self.region_feature_embedding.add(
                tf.keras.layers.Lambda(lambda x: x, name="identity")
            )
        else:
            for out_channels in mlp_region:
                self.region_feature_embedding.add(
                    tf.keras.layers.Conv2D(
                        filters=out_channels, kernel_size=(1, 1), strides=(1, 1)
                    )
                )
                if batch_norm:
                    self.region_feature_embedding.add(
                        tf.keras.layers.BatchNormalization(axis=3, momentum=0.9)
                    )
                self.region_feature_embedding.add(tf.keras.layers.LeakyReLU())

        # The output tensors.
        self._sampled_points = None
        self._grouped_features = None
        self._indices = None
        self._grouped_points = None
        self._grouped_points_norm = None

    def call(
        self,
        inputs: List[tf.Tensor],
        training: tf.Tensor = None,
        mask: tf.Tensor = None,
    ):
        """Call of PointNet Set Abstraction

        Note:
            points = inputs[0] : tf.Tensor(shape=(B, N, 3), dtype=tf.float32)
                The batched (B) xyz points with N number of points.
            features = inputs[1] : tf.Tensor(shape=(B, N, C), dtype=tf.float32)
                The batched (B) features with N data points and C channels.

        Arguments:
            inputs : list of two tensors
                Must have length 2 and contain points and features.
            training : tf.Tensor(shape=(), dtype=tf.bool)
                If `True` model is in training mode (e.g. for batch normalization).
                If `False` model is in inference mode.
            mask : tf.Tensor
                Currently unused.

        Returns:
            sampled_points : tf.Tensor(shape=(B, num_points, 3), dtype=tf.float32)
            grouped_features : tf.Tensor(shape=(B, num_points, C), dtype=tf.float32)
            indices : tf.Tensor(shape=(B, num_points, num_samples), dtype=tf.int32)
        """

        assert training is not None
        assert len(inputs) == 2, "Inputs must be `points` and `features`."
        points = inputs[0]
        features = inputs[1]

        # Sampling and Grouping.
        (
            self._sampled_points,  # (B, num_points, 3)
            grouped_features,  # (B, num_points, num_samples, C (+ 3))
            self._indices,  # (B, num_points, num_samples)
            self._grouped_points_norm,  # (B, num_points, num_samples, 3)
            self._grouped_points,  # (B, num_points, num_samples, 3)
        ) = self.sampling_and_grouping(points, features)

        # Point Feature Embedding.
        grouped_features = self.point_feature_embedding(
            grouped_features, training=training
        )

        # Pooling in Local Regions.
        grouped_features = self.pool_local_regions(
            grouped_features, self._grouped_points_norm
        )

        # Region Feature Embedding.
        grouped_features = self.region_feature_embedding(
            grouped_features, training=training
        )

        # Remove additional dimension.
        # (batch_size, num_points, 1, C) --> (batch_size, num_points, C)
        # with C = mlp_point[-1] if mlp_region is None else mlp_region[-1]
        self._grouped_features = grouped_features[:, :, 0, :]

        return self._sampled_points, self._grouped_features

    def get_query_points(self):
        return self._sampled_points

    def get_features(self):
        return self._grouped_features

    def get_neighbor_indices(self):
        return self._indices

    def get_neighborhoods(self, normalized: bool):
        if normalized:
            return self._grouped_points_norm
        else:
            return self._grouped_points
