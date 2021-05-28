#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Larissa Triess"
__email__ = "mail@triess.eu"

from typing import List, Union

import tensorflow as tf

from .modules import SetAbstractionModule


def _get_element_list(elem, levels) -> list:
    if elem is None:
        elem = [None] * levels
    elif isinstance(elem, (list, tuple)):
        if len(elem) != levels:
            raise ValueError("Length of elements must equal number of levels!")
    else:
        if not isinstance(elem, (int, float, bool, str)):
            raise TypeError("Unknown type {t}!".format(t=type(elem)))
        elem = [elem] * levels
    return elem


class PointNet2(tf.keras.models.Model):
    """PointNet++ feature extractor with set abstraction modules.

    Arguments:
        mlp_point : List[List[int]]
            A list for each abstraction module contains a list of integers
            for the output sizes of the MLP on each point.
        mlp_region : List[int]
            A list for each abstraction module contains a list of integers
            for the output sizes of the MLP on each region.
            If `None`, no further processing on the regions is conducted.
        num_queries : int or List[int]
            The number of points sampled in farthest point sampling.
        num_neighbors : int or List[int]
            The number of points sampled in each local region.
        radius : float or List[float]
            The search radius in the local region if use_knn is `False`.
        reduce : bool
            If `True`, then group all points into one point cloud. This is
            equivalent to num_points=1, radius=inf, num_samples=N with N being the
            number of points.
        use_knn : bool or List[bool]
            If `True`, use KNN search. If `False`, use radius search.
        use_xyz : bool or List[bool]
            If `True`, concat points to local point features. If `False`, only use
            local point features.
        pooling : str or List[str]
            The pooling to use to get the global from the local features.
            Must be one of `max`, `avg`, `weighted_avg`, or `max_and_avg`.
        feature_norm : str or List[str]
            The feature normalization to use. Can be `batch` for batch normalization
            or `layer` for layer normalization. If None, no normalization is applied.
        sampling : str or List[str]
            Either use Farthest or Random Point Sampling.
        coord_norm : str or List[str], must be one of `trans`, `transrot`
            Use either translation (`trans`) or translation with rotation (`transrot`)
            for the normalization of the points coordinates.
        name : str
            Name of the model.

    Raises:
        ValueError if use_knn is `False` and radius is `None`
        ValueError if group_all is `False` and one of num_points, num_samples, or
            radius is `None`

    """

    def __init__(
        self,
        mlp_point: List[List[int]],
        mlp_region: List[List[int]] = None,
        num_queries: Union[int, List[int]] = None,
        num_neighbors: Union[int, List[int]] = None,
        radius: Union[float, List[float]] = None,
        reduce: bool = True,
        use_knn: Union[bool, List[bool]] = True,
        use_xyz: Union[bool, List[bool]] = True,
        pooling: Union[str, List[str]] = "max",
        feature_norm: Union[str, List[str]] = None,
        sampling: Union[str, List[str]] = "farthest",
        coord_norm: Union[str, List[str]] = "trans",
        **_,
    ):
        super().__init__()

        self.levels = len(mlp_point)
        if not self.levels > 0:
            raise ValueError("You must provide a list of mlp units.")

        mlp_region = _get_element_list(mlp_region, levels=self.levels)
        num_queries = _get_element_list(num_queries, levels=self.levels)
        num_neighbors = _get_element_list(num_neighbors, levels=self.levels)
        radius = _get_element_list(radius, levels=self.levels)
        use_knn = _get_element_list(use_knn, levels=self.levels)
        use_xyz = _get_element_list(use_xyz, levels=self.levels)
        pooling = _get_element_list(pooling, levels=self.levels)
        feature_norm = _get_element_list(feature_norm, levels=self.levels)
        sampling = _get_element_list(sampling, levels=self.levels)
        coord_norm = _get_element_list(coord_norm, levels=self.levels)

        self.set_abstraction_layers = []
        for layer_idx in range(self.levels):
            self.set_abstraction_layers.append(
                SetAbstractionModule(
                    mlp_point=mlp_point[layer_idx],
                    mlp_region=mlp_region[layer_idx],
                    num_queries=num_queries[layer_idx],
                    num_neighbors=num_neighbors[layer_idx],
                    radius=radius[layer_idx],
                    group_all=reduce if layer_idx == self.levels - 1 else False,
                    use_knn=use_knn[layer_idx],
                    use_xyz=use_xyz[layer_idx],
                    pooling=pooling[layer_idx],
                    feature_norm=feature_norm[layer_idx],
                    sampling=sampling[layer_idx],
                    coord_norm=coord_norm[layer_idx],
                    name="SetAbstractionModule{idx:02d}".format(idx=layer_idx),
                )
            )

    def call(
        self, points: tf.Tensor, training: tf.Tensor = None, mask: tf.Tensor = None
    ) -> (tf.Tensor, dict):
        """Call of PointNet++ feature extractor

        Arguments:
            points: tf.Tensor(shape=(B, N, 3), dtype=tf.float32)
            training: tf.Tensor(shape=(), dtype=tf.bool)
            mask: tf.Tensor

        Returns:
            features: tf.Tensor(shape=(B, num_points[-1], mlp[-1][-1]), dtype=tf.float32)
            abstraction_output: Dict[List[tf.Tensor]]
        """

        features = None
        abstraction_output = {
            "features": [None],
            "queries": [points],
            "indices": [None],
            "neighborhoods": [None],
            "neighborhoods_norm": [None],
        }

        # Input: (B, N, 3) None
        # Output: (B, num_points[-1], mlp[-1][-1])
        for sa_layer in self.set_abstraction_layers:
            points, features = sa_layer(
                inputs=[points, features], training=training, mask=mask
            )

            # Add these to dict in case user wants more information.
            abstraction_output["features"].append(sa_layer.get_features())
            abstraction_output["queries"].append(sa_layer.get_query_points())
            abstraction_output["indices"].append(sa_layer.get_neighbor_indices())
            abstraction_output["neighborhoods"].append(
                sa_layer.get_neighborhoods(normalized=False)
            )
            abstraction_output["neighborhoods_norm"].append(
                sa_layer.get_neighborhoods(normalized=True)
            )

        return features, abstraction_output
