#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017.
Modified by Larissa Triess (2020)
"""

import os
import sys

import tensorflow as tf
from tensorflow.python.framework import ops

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sampling_module = tf.load_op_library(os.path.join(BASE_DIR, "tf_sampling_so.so"))


def farthest_point_sample(k: int, points: tf.Tensor) -> tf.Tensor:
    """Returns the indices of the k farthest points in points

    Arguments:
        k : int
            The number of points to consider.
        points : tf.Tensor(shape=(batch_size, P1, 3), dtype=tf.float32)
            The points with P1 dataset points given in xyz.

    Returns:
        indices : tf.Tensor(shape=(batch_size, k), dtype=tf.int32)
            The indices of the k farthest points in points.

    """

    return sampling_module.farthest_point_sample(points, k)


ops.NoGradient("FarthestPointSample")
