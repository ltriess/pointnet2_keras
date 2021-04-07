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
In your project, you may use the feature extractor from this implementation.
We do not provide the implementation of the output heads.

```python
import tensorflow as tf
from pointnet2 import PointNet2

feature_extractor = PointNet2(
    mlp_point=[[64, 64, 128], [128, 128, 256]],
    num_queries=[100, 10],
    num_neighbors=[10, 10],
    radius=[0.2, 0.4],
    use_knn=False,
    use_xyz=True,
    sampling="random",
)

classifier = ...

points = tf.random.normal(shape=(2, 1000, 3))
features, _ = feature_extractor(points)
predictions = classifier(features)
```

### License
This code is released under MIT License (see [LICENSE](LICENSE) for details).
This code is partially adapted from [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2), as indicated in each affected file.
