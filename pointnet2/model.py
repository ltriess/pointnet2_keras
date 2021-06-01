#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Larissa Triess"
__email__ = "mail@triess.eu"

from typing import Dict, List, Union

import tensorflow as tf

from .modules import FeaturePropagationModule, SetAbstractionModule


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


class Classifier(tf.keras.models.Model):
    """PointNet++ Classifier

    This classifier is used for the normal version of the feature extractor and
    the Multi-Scale Grouping (MSG) version as well.
    The dropout ratios differ a bit. For MSG it's 0.6, for the normal version it's 0.5.

    Arguments:
        units : List[float]
            A list of integers for the unit sizes of the dense layers.
            Paper default is [512, 256, 40].
        dropout_rate : float or List[float]
            The dropout ratio applied after each dense layer. Paper default is 0.5~0.6.
            If None, no dropout will be used.
        feature_norm : str or List[str]
            The feature normalization to use. Can be `batch` for batch normalization or
            `layer` for layer normalization. If None, no normalization is applied.
            Paper default is `batch`.

    Raises:
        ValueError if feature_norm is not one of `batch`, `layer`, or None.

    """

    def __init__(
        self,
        units: List[int],
        dropout_rate: Union[float, List[float]] = 0.5,
        feature_norm: Union[str, List[str]] = None,
    ):
        super().__init__()

        self.levels = len(units)

        dropout_rate = _get_element_list(dropout_rate, levels=self.levels)
        feature_norm = _get_element_list(feature_norm, levels=self.levels)

        if not all(n in {None, "batch", "layer"} for n in feature_norm):
            raise ValueError("Normalization can only be `None`, `batch` or `layer`!")

        self.model = tf.keras.models.Sequential(name="PointNet2Classifier")

        for i in range(self.levels - 1):
            self.model.add(tf.keras.layers.Dense(units[i]))
            if feature_norm[i] == "batch":
                self.model.add(tf.keras.layers.BatchNormalization())
            elif feature_norm[i] == "layer":
                self.model.add(tf.keras.layers.LayerNormalization())
            else:
                pass
            self.model.add(tf.keras.layers.LeakyReLU())
            if dropout_rate[i] is not None:
                self.model.add(tf.keras.layers.Dropout(rate=dropout_rate[i]))
        self.model.add(tf.keras.layers.Dense(units[-1]))

    def call(
        self, features: tf.Tensor, training: tf.Tensor = None, mask: tf.Tensor = None
    ) -> tf.Tensor:
        """Call of PointNet++ classifier

        Arguments:
            features: tf.Tensor(shape=(B, *, C), dtype=tf.float32)
                For example feature output of PointNet++ feature extractor.
            training: tf.Tensor(shape=(), dtype=tf.bool)
            mask: tf.Tensor

        Returns:
            logits: tf.Tensor(shape=(B, *, units[-1]), dtype=tf.float32)
        """

        return self.model(features, training=training, mask=mask)


class SegmentationModel(tf.keras.models.Model):
    """PointNet++ feature extractor with set abstraction modules.

    Arguments:
        fp_units : List[List[int]]
            A list for each feature propagation contains a list of integers
            for the output sizes of the MLP on each point. Must contain same or less
            amount of layers than the feature extractor.
        num_classes: int
            The number of classes to make predictions for.
        dropout_rate : float or List[float]
            The dropout ratio applied in each block. Paper default is 0.5.
            If None, no dropout will be used.
        feature_norm : str
            The feature normalization to use. Can be `batch` for batch normalization
            or `layer` for layer normalization. If None, no normalization is applied.

    Raises:
        ValueError if feature normalization is not valid.

    """

    def __init__(
        self,
        fp_units: List[List[int]],
        num_classes: int,
        dropout_rate: float = 0.5,
        feature_norm: str = None,
    ):
        super().__init__()

        self.levels = len(fp_units)

        if feature_norm not in {None, "batch", "layer"}:
            raise ValueError(f"Received unknown feature norm `{feature_norm}`!")

        self.segmentation_layers = [
            FeaturePropagationModule(mlp_units=units, feature_norm=feature_norm)
            for units in fp_units
        ]

        self.head = tf.keras.models.Sequential(name="head")
        self.head.add(tf.keras.layers.Conv1D(128, kernel_size=1, padding="valid"))
        if feature_norm == "batch":
            self.head.add(tf.keras.layers.BatchNormalization())
        elif feature_norm == "layer":
            self.head.add(tf.keras.layers.LayerNormalization())
        else:
            pass
        self.head.add(tf.keras.layers.LeakyReLU())
        if dropout_rate is not None:
            self.head.add(tf.keras.layers.Dropout(rate=dropout_rate))
        self.head.add(
            tf.keras.layers.Conv1D(num_classes, kernel_size=1, padding="valid")
        )

    def call(
        self,
        abstraction_output: Dict[str, List[tf.Tensor]],
        training: tf.Tensor = None,
        mask: tf.Tensor = None,
    ) -> tf.Tensor:
        """Call of PointNet++ segmentation model

        Arguments:
            abstraction_output: Dict[List[tf.Tensor[dtype=tf.float32]]
                The abstraction output dict of PointNet++ feature extractor.
            training: tf.Tensor(shape=(), dtype=tf.bool)
            mask: tf.Tensor

        Returns:
            logits: tf.Tensor(shape=(B, N, num_classes), dtype=tf.float32)
        """

        if "queries" not in abstraction_output:
            raise KeyError("Dict must contain key `queries`!")
        if "features" not in abstraction_output:
            raise KeyError("Dict must contain key `features`!")
        if not len(abstraction_output["features"]) == len(
            abstraction_output["queries"]
        ):
            raise RuntimeError("Inconsistent feature and query entries in dict!")
        if not len(abstraction_output["features"]) >= self.levels:
            raise RuntimeError("Received less features than levels in the model!")

        # Feature Propagation Layers.
        features = abstraction_output["features"][-1]
        for i in range(self.levels):

            features = self.segmentation_layers[i](
                inputs=[
                    abstraction_output["queries"][-i - 2],
                    abstraction_output["queries"][-i - 1],
                    abstraction_output["features"][-i - 2],
                    features,
                ],
                training=training,
                mask=mask,
            )

        # FC Layers.
        logits = self.head(features, training=training, mask=mask)
        return logits


class FeatureExtractor(tf.keras.models.Model):
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
