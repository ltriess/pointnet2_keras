#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Larissa Triess"
__email__ = "mail@triess.eu"


from typing import List

import tensorflow as tf


class RandomPointSampling(tf.keras.layers.Layer):
    def __init__(self, num_queries):
        """ Randomly sample query points from list of points """
        super().__init__()

        self.num_queries = num_queries

    def call(self, inputs: List[tf.Tensor], **_) -> tf.Tensor:
        points = inputs[0]

        # Create indices for one batch element.
        indices = tf.range(start=0, limit=tf.shape(points)[1], dtype=tf.int32)
        # Broadcast the indices to the complete mini batch.
        indices = tf.broadcast_to(indices[tf.newaxis, ...], tf.shape(points)[:-1])
        check = tf.assert_equal(tf.shape(points)[:-1], tf.shape(indices))
        # Use map function to apply different shuffles to each batch indices.
        with tf.control_dependencies([check]):
            indices = tf.map_fn(lambda i: tf.random.shuffle(i), indices)
        # Take the first M entries for random indices.
        indices = indices[:, : self.num_queries]

        return indices
