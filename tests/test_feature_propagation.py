#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Larissa Triess"
__email__ = "mail@triess.eu"


import tensorflow as tf

from pointnet2.modules.feature_propagation import FeaturePropagationModule


class TestModelUsage(tf.test.TestCase):
    def setUp(self):
        self.points_hr = tf.random.normal((2, 400, 3))
        self.points_lr = tf.random.normal((2, 100, 3))
        self.features_hr = tf.random.normal((2, 400, 8))
        self.features_lr = tf.random.normal((2, 100, 32))

    def test_output_shape_with_hr_features(self):
        fp = FeaturePropagationModule(mlp_units=[32, 64])
        new_features = fp(
            [self.points_hr, self.points_lr, self.features_hr, self.features_lr]
        )
        self.assertEqual((2, 400, 64), new_features.shape)

    def test_output_shape_without_hr_features(self):
        fp = FeaturePropagationModule(mlp_units=[32, 64])
        new_features = fp([self.points_hr, self.points_lr, None, self.features_lr])
        self.assertEqual((2, 400, 64), new_features.shape)


if __name__ == "__main__":
    tf.test.main()
