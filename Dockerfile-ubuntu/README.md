# Build Dockerfile for Intel® End-to-End AI Optimization Kit

## How to build it

### By default, you can build e2eAIOK docker image with latest nightly-build e2eAIOK PyPI distributions.
``` bash
$ cd Dockerfile-ubuntu
$ docker build -t e2eaiok-tensorflow . -f DockerfileTensorflow
$ docker build -t e2eaiok-pytorch . -f DockerfilePytorch
$ docker build -t e2eaiok-pytorch112 . -f DockerfilePytorch112
```

### If you need http and https proxy to build docker image:
``` bash
$ cd Dockerfile-ubuntu
$ docker build -t e2eaiok-tensorflow . -f DockerfileTensorflow --build-arg http_proxy=http://proxy-ip:proxy-port --build-arg https_proxy=http://proxy-ip:proxy-port
$ docker build -t e2eaiok-pytorch . -f DockerfilePytorch --build-arg http_proxy=http://proxy-ip:proxy-port --build-arg https_proxy=http://proxy-ip:proxy-port
$ docker build -t e2eaiok-pytorch112 . -f DockerfilePytorch112 --build-arg http_proxy=http://proxy-ip:proxy-port --build-arg https_proxy=http://proxy-ip:proxy-port
```

## How to use the image.

### Option 1: To start a notebook directly with a specified port(e.g. 12888).
``` bash
docker run -it --rm -p 12888:12888 e2eaiok-tensorflow:latest
docker run -it --rm -p 12888:12888 e2eaiok-pytorch:latest
docker run -it --rm -p 12888:12888 e2eaiok-pytorch112:latest
```

### Option 2: To start a cluster via one-click script
``` bash
git clone https://github.com/intel/e2eAIOK.git
cd e2eAIOK
git submodule update --init --recursive
```

#### Option 2.1: Start a cluster w/o jupyter notebook service enabled
``` bash
python scripts/start_e2eaiok_docker.py --backend [tensorflow, pytorch, pytorch112] --dataset_path ../ --workers host1, host2, host3, host4 --proxy "http://addr:ip"
```

#### Option 2.2: Start a cluster w/ jupyter notebook service enabled
``` bash
python scripts/start_e2eaiok_docker.py --backend [tensorflow, pytorch, pytorch112] --dataset_path ../ --workers host1, host2, host3, host4 --proxy "http://addr:ip" --jupyter_mode
```

## Versions and Components

### DockerfilePytorch
* PyTorch 1.5
* Intel® Extension for Pytorch 0.2, 1.12.x
* 3.7.16

### DockerfilePytorch112
* PyTorch 1.12
* Intel® Extension for Pytorch 1.12.x
* Python 3.9.15

### DockerfileTensorflow
* TensorFlow 2.10.0
* Intel® Extension for TensorFlow 2.10.x
* Horovod 0.26
* Python 3.9.12
