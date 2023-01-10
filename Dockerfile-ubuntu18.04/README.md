## Build Dockerfile for e2eAIOK workloads

```
$ cd Dockerfile-ubuntu18.04
$ docker build -t e2eaiok-tensorflow . -f DockerfileTensorflow
$ docker build -t e2eaiok-pytorch . -f DockerfilePytorch
$ docker build -t e2eaiok-pytorch110 . -f DockerfilePytorch110
$ docker build -t e2eaiok-pytorch112 . -f DockerfilePytorch112
```

Notice:
If you need a proxy to build docker, please use below build scripts instead.
```
$ cd Dockerfile-ubuntu18.04
$ docker build -t e2eaiok-tensorflow . -f DockerfileTensorflow --build-arg http_proxy=http://proxy-ip:proxy-port --build-arg https_proxy=http://proxy-ip:proxy-port
$ docker build -t e2eaiok-pytorch . -f DockerfilePytorch --build-arg http_proxy=http://proxy-ip:proxy-port --build-arg https_proxy=http://proxy-ip:proxy-port
$ docker build -t e2eaiok-pytorch110 . -f DockerfilePytorch110 --build-arg http_proxy=http://proxy-ip:proxy-port --build-arg https_proxy=http://proxy-ip:proxy-port
$ docker build -t e2eaiok-pytorch112 . -f DockerfilePytorch112 --build-arg http_proxy=http://proxy-ip:proxy-port --build-arg https_proxy=http://proxy-ip:proxy-port
```
