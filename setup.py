# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="pointnet2",
    version="0.1.0",
    author="Larissa Triess",
    author_email="mail@triess.eu",
    description="Tensorflow 1 Keras implementation of PointNet++ with some additions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ltriess/pointnet2_keras",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.6",
    extras_require={"tf-cpu": ["tensorflow==1.15"], "tf-gpu": ["tensorflow-gpu==1.15"]},
)
