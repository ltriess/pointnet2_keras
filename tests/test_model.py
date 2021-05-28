#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Larissa Triess"
__email__ = "mail@triess.eu"


import tensorflow as tf

from pointnet2 import PointNet2


class TestModelUsage(tf.test.TestCase):
    def setUp(self):
        self.mlp_point = [[64, 128], [128, 256], [256, 512]]
        self.num_queries = [100, 10, -1]
        self.num_neighbors = [10, 10, -1]

    def test_inputs_01(self):
        model = PointNet2(
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
            PointNet2,
            mlp_point=self.mlp_point,
            num_queries=num_queries,
            num_neighbors=num_neighbors,
            reduce=True,
        )

    def test_inputs_03(self):
        mlp_point = [[64, 128], [128, 256]]
        num_queries = [100, 10]
        num_neighbors = [10, 10]

        model = PointNet2(
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

        model = PointNet2(
            mlp_point=mlp_point,
            num_queries=num_queries,
            num_neighbors=num_neighbors,
            reduce=False,
        )

        points = tf.random.normal((2, 1000, 3))
        features, _ = model(points, training=tf.constant(False))
        self.assertEqual((2, 10, 256), features.shape)

    def test_normalization_param(self):
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            coord_norm="transrot",
        )
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            coord_norm="trans",
        )
        self.assertRaises(
            ValueError,
            PointNet2,
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            coord_norm="something",
        )

    def test_sampling_param(self):
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            sampling="random",
        )
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            sampling="farthest",
        )
        self.assertRaises(
            ValueError,
            PointNet2,
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            sampling="something",
        )

    def test_feature_norm_param(self):
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            feature_norm="batch",
        )
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            feature_norm="layer",
        )
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            feature_norm=None,
        )
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            feature_norm=["batch", "layer", None],
        )
        self.assertRaises(
            ValueError,
            PointNet2,
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            feature_norm=[True, False],
        )
        self.assertRaises(
            ValueError,
            PointNet2,
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            feature_norm=["test", None, None],
        )
        self.assertRaises(
            ValueError,
            PointNet2,
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            feature_norm="R2D2",
        )

    def test_xyz_usage_param(self):
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            use_xyz=True,
        )
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            use_xyz=False,
        )
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            use_xyz=[False, False, True],
        )

    def test_knn_radius_search(self):
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            use_knn=True,
        )
        self.assertRaises(
            ValueError,
            PointNet2,
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            use_knn=False,
        )
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            radius=0.2,
            use_knn=False,
        )
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            radius=[0.2, 0.4, 0.4],
            use_knn=False,
        )
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            radius=0.4,
            use_knn=[True, False, True],
        )
        PointNet2(
            mlp_point=self.mlp_point,
            num_queries=self.num_queries,
            num_neighbors=self.num_neighbors,
            radius=[None, 0.6, -1],
            use_knn=[True, False, True],
        )


if __name__ == "__main__":
    tf.test.main()
