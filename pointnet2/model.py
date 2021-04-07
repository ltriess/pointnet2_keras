#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Larissa Triess"
__email__ = "mail@triess.eu"

from typing import List, Union

import tensorflow as tf

from .modules import SetAbstractionModule


class PointNet2(tf.keras.models.Model):
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
        batch_norm: Union[bool, List[bool]] = True,
        sampling: Union[str, List[str]] = "farthest",
        normalization: Union[str, List[str]] = "trans",
        **_,
    ):
        super().__init__()
        """PointNet++ feature extractor with set abstraction modules.

        Arguments:
            mlp_point : List[List[int]]
                A list for each abstraction module contains a list of integers
                for the output sizes of the MLP on each point.
            mlp_region : List[int]
                A list for each abrtaction module contains a list of integers
                for the output sizes of the MLP on each region.
                If `None`, no further processing on the regions is conducted.
            num_queries : int
                The number of points sampled in farthest point sampling.
            num_neighbors : int
                The number of points sampled in each local region.
            radius : float
                The search radius in the local region if use_knn is `False`.
            reduce : bool
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

        self.levels = len(mlp_point)
        if not self.levels > 0:
            raise ValueError("You must provide a list of mlp units.")

        mlp_region = self.get_element_list(mlp_region)
        num_queries = self.get_element_list(num_queries)
        num_neighbors = self.get_element_list(num_neighbors)
        radius = self.get_element_list(radius)
        use_knn = self.get_element_list(use_knn)
        use_xyz = self.get_element_list(use_xyz)
        pooling = self.get_element_list(pooling)
        batch_norm = self.get_element_list(batch_norm)
        sampling = self.get_element_list(sampling)
        normalization = self.get_element_list(normalization)

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
                    batch_norm=batch_norm[layer_idx],
                    sampling=sampling[layer_idx],
                    normalization=normalization[layer_idx],
                    name="SetAbstractionModule{idx:02d}".format(idx=layer_idx),
                )
            )

    def get_element_list(self, elem) -> list:
        if elem is None:
            elem = [None] * self.levels
        elif isinstance(elem, (list, tuple)):
            if len(elem) != self.levels:
                raise ValueError("Length of elements must equal number of levels!")
        else:
            if not isinstance(elem, (int, float, bool, str)):
                raise TypeError("Unknown type {t}!".format(t=type(elem)))
            elem = [elem] * self.levels
        return elem

    def call(
        self, points: tf.Tensor, training: tf.Tensor = None, mask: tf.Tensor = None
    ) -> (tf.Tensor, dict):

        features = None
        abstraction_output = {
            "features": [],
            "queries": [],
            "indices": [],
            "neighborhoods": [],
            "neighborhoods_norm": [],
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
