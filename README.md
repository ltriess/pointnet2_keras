## TF1 Keras implementation for PointNet++ Model
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![version](https://img.shields.io/badge/version-0.1.0-blue)

This repository holds the implementation of the PointNet++ Model.
There is no training code contained, it is simply to use the model as a place-in feature extractor in other projects.

The original PointNet++ code can be found [here](https://github.com/charlesq34/pointnet2).
It also provides the entire training pipeline for PointNet++.

### Installation
The code requires Python 3.6 and TensorFlow 1.15 GPU version.
```commandline
pip install git+https://github.com/ltriess/pointnet2_keras
```
If you want to install TF15 alongside, use
```commandline
pip install git+https://github.com/ltriess/pointnet2_keras[tf-gpu]  # for gpu support
pip install git+https://github.com/ltriess/pointnet2_keras[tf-cpu]  # for cpu support
```

#### Compile Customized TF Operators
The TF operators are included under `tf_ops`.
Check `tf_xxx_compile.sh` under each ops subfolder to compile the operators.
The scripts are tested under TF1.15.

First, find TensorFlow include and library paths.
```commandline
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
```
Then, build the op (name indicated as `xxx`) located in each subfolder.
```commandline
/usr/local/cuda/bin/nvcc -std=c++11 -c -o tf_xxx_g.cu.o tf_xxx_g.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared -o tf_xxx_so.so tf_xxx.cpp tf_xxx_g.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -I/usr/local/cuda/include -L/usr/local/cuda/lib64
```

This implementation does not provide the nearest neighbor op.
The original implementation is not suited to deal with very large point clouds (the target of our work).
However, we cannot provide our implementation of the efficient KNN op.
You can plug in your custom TF ops `my_tf_ops` and use your KNN implementation in `layers/sample_and_group.py`
(refer to call `from my_tf_ops.knn_op import k_nearest_neighbor_op as get_knn`).

### Usage
This repo provides the implementation of the PointNet++ model without any additional code.
In your project, you can use the feature extractor and the classification and segmentation heads from this implementation independently.

#### Classification
```python
import tensorflow as tf
from pointnet2 import Classifier, FeatureExtractor

feature_extractor = FeatureExtractor(
    mlp_point=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
    num_queries=[512, 128, None],
    num_neighbors=[32, 64, None],
    radius=[0.2, 0.4, None],
    reduce=True,
    use_knn=False,
)

classifier = Classifier(units=[256, 128, 40], dropout_rate=0.4)

points = tf.random.normal(shape=(2, 2048, 3))
features, _ = feature_extractor(points)  # [2, 1024]
classification_predictions = classifier(features)  # [2, 40]
```

#### Segmentation
```python
import tensorflow as tf
from pointnet2 import FeatureExtractor, SegmentationModel

feature_extractor = FeatureExtractor(
    mlp_point=[[32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 512]],
    num_queries=[1024, 256, 64, 16],
    num_neighbors=[32, 32, 32, 32],
    radius=[0.1, 0.2, 0.4, 0.8],
    reduce=False,
    use_knn=False,
)

segmentation_model = SegmentationModel(
    fp_units=[[256, 256], [256, 256], [256, 128, 128]], num_classes=12
)

points = tf.random.normal(shape=(2, 2048, 3))
_, abstraction_output = feature_extractor(points)  # [2, 16, 512]
segmentation_predictions = segmentation_model(abstraction_output)  # [4, 2048, 12]
```

### License
This code is released under MIT License (see [LICENSE](LICENSE) for details).
This code is adapted from [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2), as indicated in each affected file.
