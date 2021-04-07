#!/bin/bash
## AUTHOR: Larissa Triess
## tested with TF1.15 and cuda-10.0

# make sure to source the correct python virtualenv first, otherwise no or the wrong tensorflow will be found
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
# should look something like this, check the paths first
# TF_CFLAGS: -I/path/to/venv/lib/python3.6/site-packages/tensorflow_core/include -D_GLIBCXX_USE_CXX11_ABI=0
# TF_LFLAGS: -L/path/to/venv/lib/python3.6/site-packages/tensorflow_core -l:libtensorflow_framework.so.1

# symlink is set to cuda version (/usr/local/cuda --> /usr/local/cuda-10.0)
/usr/local/cuda/bin/nvcc -std=c++11 -c -o tf_sampling_g.cu.o tf_sampling_g.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared -o tf_sampling_so.so tf_sampling.cpp tf_sampling_g.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -I/usr/local/cuda/include -L/usr/local/cuda/lib64
