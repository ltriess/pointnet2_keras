#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Larissa Triess"
__email__ = "mail@triess.eu"


import tensorflow as tf

from pointnet2 import FeatureExtractor, SegmentationModel


class TestModelUsage(tf.test.TestCase):
    def setUp(self):
        self.mlp_point = [[32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 512]]
        self.num_queries = [1024, 256, 64, 16]
        self.num_neighbors = [32, 32, 32, 32]
        self.radius = [0.1, 0.2, 0.4, 0.8]

        self.feature_extractor = FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            radius=self.radius,
            reduce=False,
            use_knn=False,
            feature_norm="batch",
        )

    def test_paper_default(self):
        fp_units = [[256, 256], [256, 256], [256, 128], [128, 128, 128]]
        num_classes = 12

        segmentation_model = SegmentationModel(
            fp_units=fp_units, num_classes=num_classes, feature_norm="batch"
        )

        points = tf.random.uniform((4, 2048, 3), minval=-1, maxval=1)
        features, abstraction_output = self.feature_extractor(
            points, training=tf.constant(False)
        )
        logits = segmentation_model(abstraction_output, training=tf.constant(False))

        self.assertEqual((4, 16, 512), features.shape)
        self.assertEqual((4, 2048, num_classes), logits.shape)

    def test_partial_segmentation_01(self):
        fp_units = [[256, 256], [256, 256], [256, 128, 128]]
        num_classes = 12

        segmentation_model = SegmentationModel(
            fp_units=fp_units, num_classes=num_classes, feature_norm="batch"
        )

        points = tf.random.uniform((4, 2048, 3), minval=-1, maxval=1)
        features, abstraction_output = self.feature_extractor(
            points, training=tf.constant(False)
        )
        logits = segmentation_model(abstraction_output, training=tf.constant(False))

        self.assertEqual((4, 16, 512), features.shape)
        self.assertEqual((4, self.num_queries[0], num_classes), logits.shape)

    def test_partial_segmentation_02(self):
        fp_units = [[256, 256], [256, 128, 128]]
        num_classes = 12

        segmentation_model = SegmentationModel(
            fp_units=fp_units, num_classes=num_classes, feature_norm="batch"
        )

        points = tf.random.uniform((4, 2048, 3), minval=-1, maxval=1)
        features, abstraction_output = self.feature_extractor(
            points, training=tf.constant(False)
        )
        logits = segmentation_model(abstraction_output, training=tf.constant(False))

        self.assertEqual((4, 16, 512), features.shape)
        self.assertEqual((4, self.num_queries[1], num_classes), logits.shape)


class TestFeatureExtractorUsage(tf.test.TestCase):
    def setUp(self):
        self.mlp_point = [[64, 128], [128, 256], [256, 512]]
        self.num_queries = [100, 10, -1]
        self.num_neighbors = [10, 10, -1]

    def test_inputs_01(self):
        model = FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            reduce=True,
            use_knn=True,
            use_xyz=True,
            layer_norm=None,
        )

        points = tf.random.normal((2, 1000, 3))
        features, _ = model(points, training=tf.constant(False))
        self.assertEqual((2, 1, 512), features.shape)

    def test_inputs_02(self):
        num_queries = [100, 10]
        num_neighbors = [10, 10]

        self.assertRaises(
            ValueError,
            FeatureExtractor,
            mlp_point=self.mlp_point,
            num_queries=num_queries,
            num_neighbors=num_neighbors,
            reduce=True,
        )

    def test_inputs_03(self):
        mlp_point = [[64, 128], [128, 256]]
        num_queries = [100, 10]
        num_neighbors = [10, 10]

        model = FeatureExtractor(
            mlp_point=mlp_point,
            num_queries=num_queries,
            num_neighbors=num_neighbors,
            reduce=True,
        )

        points = tf.random.normal((2, 1000, 3))
        features, _ = model(points, training=tf.constant(False))
        self.assertEqual((2, 1, 256), features.shape)

    def test_inputs_04(self):
        mlp_point = [[64, 128], [128, 256]]
        num_queries = [100, 10]
        num_neighbors = [10, 10]

        model = FeatureExtractor(
            mlp_point=mlp_point,
            num_queries=num_queries,
            num_neighbors=num_neighbors,
            reduce=False,
        )

        points = tf.random.normal((2, 1000, 3))
        features, _ = model(points, training=tf.constant(False))
        self.assertEqual((2, 10, 256), features.shape)

    def test_normalization_param(self):
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            coord_norm="transrot",
        )
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            coord_norm="trans",
        )
        self.assertRaises(
            ValueError,
            FeatureExtractor,
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            coord_norm="something",
        )

    def test_sampling_param(self):
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            sampling="random",
        )
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            sampling="farthest",
        )
        self.assertRaises(
            ValueError,
            FeatureExtractor,
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            sampling="something",
        )

    def test_feature_norm_param(self):
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            feature_norm="batch",
        )
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            feature_norm="layer",
        )
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            feature_norm=None,
        )
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            feature_norm=["batch", "layer", None],
        )
        self.assertRaises(
            ValueError,
            FeatureExtractor,
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            feature_norm=[True, False],
        )
        self.assertRaises(
            ValueError,
            FeatureExtractor,
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            feature_norm=["test", None, None],
        )
        self.assertRaises(
            ValueError,
            FeatureExtractor,
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            feature_norm="R2D2",
        )

    def test_xyz_usage_param(self):
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            use_xyz=True,
        )
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            use_xyz=False,
        )
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            use_xyz=[False, False, True],
        )

    def test_knn_radius_search(self):
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            use_knn=True,
        )
        self.assertRaises(
            ValueError,
            FeatureExtractor,
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            use_knn=False,
        )
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            radius=0.2,
            use_knn=False,
        )
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            radius=[0.2, 0.4, 0.4],
            use_knn=False,
        )
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            radius=0.4,
            use_knn=[True, False, True],
        )
        FeatureExtractor(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            radius=[None, 0.6, -1],
            use_knn=[True, False, True],
        )


if __name__ == "__main__":
    tf.test.main()
