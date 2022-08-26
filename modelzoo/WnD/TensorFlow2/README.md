# Intel Optimized Wide and Deep

## pre-work
sync submodule code
```
git submodule update --init --recursive
```

apply patch
```
cd modelzoo/WnD/TensorFlow2
bash patch_wnd.patch
```

Source repo: https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep

## Model

Google's [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)

## Environment setup

* Spark data preprocess: install [Spark](https://spark.apache.org)

* Tensorflow

  ```
  pip install intel-tensorflow
  ```

* oneCCL (optional)

  Follow [oneCCL](https://github.com/oneapi-src/oneCCL) installation guide

* OpenMPI (optional)

  Note that oneCCL contains Intel mpi, so if you installed oneCCL, the installation for OpenMPI is not needed.

  Follow [OpenMPI](https://www.open-mpi.org/faq/?category=building) installation guide to build OpenMPI

* Horovod

  ```
  HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir horovod[intel-tensorflow]
  ```

* Tensorflow transform

  ```
  pip install --no-cache-dir tensorflow-transform==0.24.1 tensorflow-metadata==0.14.0 pydot dill
  ```

* or pull the docker image `xuechendi/oneapi-aikit:hydro.ai`

## Dataset

The original dataset can be downloaded at https://www.kaggle.com/c/outbrain-click-prediction/data

## Quick start guide

### Data preprocessing

`bash scripts/spark_preproc.sh`

### Training

`bash scripts/train.sh`

### Inference

`bash scripts/inference.sh`
