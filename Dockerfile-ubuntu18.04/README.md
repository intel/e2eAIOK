## Build Dockerfile for e2eAIOK workloads

```
$ cd Dockerfile-ubuntu18.04
$ docker build -t e2eaiok-tensorflow . -f DockerfileTensorflow
$ docker build -t e2eaiok-pytorch . -f DockerfilePytorch
```

Notice:
If you need a proxy to build docker, please use below build scripts instead.
```
$ cd Dockerfile-ubuntu18.04
$ docker build -t e2eaiok-tensorflow . -f DockerfileTensorflow --build-arg http_proxy --build-arg https_proxy
$ docker build -t e2eaiok-pytorch . -f DockerfilePytorch --build-arg http_proxy --build-arg https_proxy
```


