#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2017 Charles R. Qi
Modified by Larissa Triess (2020)
"""

import os
import sys

import tensorflow as tf
from tensorflow.python.framework import ops

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
grouping_module = tf.load_op_library(os.path.join(BASE_DIR, "tf_grouping_so.so"))


def query_ball_point(
    radius: float, k: int, xyz1: tf.Tensor, xyz2: tf.Tensor
) -> (tf.Tensor, tf.Tensor):
    """Return k points within a ball region of radius around the query points.

    Arguments:
        radius : float
            The ball region search radius.
        k : int
            The number of points selected in each ball region.
        xyz1 : tf.Tensor(shape=(batch_size, P1, 3), dtype=tf.float32)
            The input points with P1 number of points given in xyz.
        xyz2 : tf.Tensor(shape=(batch_size, P2, 3), dtype=tf.float32)
            The query points with P2 number of points given in xyz.

    Returns:
        indices : tf.Tensor(shape=(batch_size, P2, k), dtype=tf.int32)
            The indices of the k ball region points in the input points.
        unique_point_count : tf.Tensor(shape=(batch_size, k), dtype=tf.int32)
            The number of unique points in each local region.
    """
    return grouping_module.query_ball_point(xyz1, xyz2, radius, k)


ops.NoGradient("QueryBallPoint")


def select_top_k(k: int, data: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """Returns the indices and elements of the k smallest elements of data.

    Arguments:
        k : int
            The number of the k SMALLEST elements selected.
        data : tf.Tensor(shape=(batch_size, m, n), dtype=tf.float32)
            A distance matrix with m query points and n dataset points.

    Returns:
        indices : tf.Tensor(shape=(batch_size, m, n), dtype=tf.int32)
            The first k in n are the indices to the top k.
        distances : tf.Tensor(shape=(batch_size, m, n), dtype=tf.float32)
            The first k in n are the distances of the top k.
    """

    return grouping_module.selection_sort(data, k)


ops.NoGradient("SelectionSort")


def group_point(data: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
    """Group points according to indices.

    Arguments:
        data : tf.Tensor(shape=(batch_size, P1, channels), dtype=tf.float32)
            The data to sample from with P1 number of points.
        indices : tf.Tensor(shape=(batch_size, P2, k), dtype=tf.int32)
            The indices to the points with P2 query positions and k entries.

    Returns:
        group : tf.Tensor(shape=(batch_size, P2, k, channels), dtype=tf.float32)
            The values sampled from points with indices.

    """

    return grouping_module.group_point(data, indices)


@tf.RegisterGradient("GroupPoint")
def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [grouping_module.group_point_grad(points, idx, grad_out), None]


def knn_point(k: int, xyz1: tf.Tensor, xyz2: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """Compute the distances and indices of the k nearest neighbors.

    Arguments:
        k : int
            The number of neighbors to return in KNN search.
        xyz1 : tf.Tensor(shape=(batch_size, P1, channels), dtype=tf.float32)
            The input points with P1 number of points and channels given in xyz.
        xyz2 : tf.Tensor(shape=(batch_size, P2, channels), dtype=tf.float32)
            The query points with P2 number of points and channels given in xyz.

    Returns:
        indices : tf.Tensor(shape=(batch_size, P2, k), dtype=tf.int32)
            The indices of of the k nearest neighbors in the input points.
        distances : tf.Tensor(shape=(batch_size, P2, k), dtype=tf.float32)
            The L2 distances of the k nearest neighbors.
    """
    # Internal broadcast to (batch_size, num_points_2, num_points_1, channels).
    xyz1 = xyz1[:, tf.newaxis, :, :]
    xyz2 = xyz2[:, :, tf.newaxis, :]

    # Compute the L2 distances.
    distances = tf.reduce_sum((xyz1 - xyz2) ** 2, axis=-1)

    # Use select_to_k with GPU support (tf.nn.top_k only has CPU support).
    indices, distances = select_top_k(k, distances)
    indices = indices[..., :k]
    distances = distances[..., :k]

    return indices, distances
