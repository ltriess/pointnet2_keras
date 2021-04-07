#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Larissa Triess"
__email__ = "mail@triess.eu"


import tensorflow as tf


class MaxPool(tf.keras.layers.Layer):
    def __init__(self):
        super(MaxPool, self).__init__()

    def call(self, *args):
        features = args[0]  # (batch_size, num_points, num_samples, channels)
        return tf.reduce_max(features, axis=2, keepdims=True, name=self.name)


class AvgPool(tf.keras.layers.Layer):
    def __int__(self):
        super(AvgPool, self).__init__()

    def call(self, *args):
        features = args[0]  # (batch_size, num_points, num_samples, channels)
        return tf.reduce_mean(features, axis=2, keepdims=True, name=self.name)


class WeightedAvgPool(tf.keras.layers.Layer):
    def __init__(self):
        super(WeightedAvgPool, self).__init__()

    def call(self, *args):
        assert len(args) == 2
        features = args[0]  # (batch_size, num_points, num_samples, channels)
        points = args[1]  # (batch_size, num_points, num_samples, 3)

        # Compute the weights from the points by euclidean distance weighting.
        # distances shape (batch_size, num_points, num_samples, 1)
        distances = tf.norm(points, axis=-1, ord=2, keepdims=True)
        exp_dists = tf.exp(-distances * 5)
        # weights shape (batch_size, num_points, num_samples, 1)
        weights = tf.div_no_nan(
            exp_dists, tf.reduce_sum(exp_dists, axis=2, keepdims=True)
        )

        # Compute the weighted sum.
        # (batch_size, num_points, num_samples, channels) --> (batch_size, num_points, 1, channels)
        features *= weights
        features = tf.reduce_sum(features, axis=2, keepdims=True)

        return features


class MaxAndAvgPool(tf.keras.layers.Layer):
    def __init__(self):
        super(MaxAndAvgPool, self).__init__()
        self.max_pool = MaxPool()
        self.avg_pool = AvgPool()

    def call(self, *args):
        features = args[0]  # (batch_size, num_points, num_samples, channels)
        avg_features = self.avg_pool(features)
        max_features = self.max_pool(features)
        features = tf.concat([avg_features, max_features], axis=-1)
        return features  # (batch_size, num_points, num_samples, 2 * channels)
