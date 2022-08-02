## Build Dockerfile for AIDK workloads

```
$ cd Dockerfile-ubuntu18.04
$ docker build -t aidk-tensorflow . -f DockerfileTensorflow
$ docker build -t aidk-pytorch . -f DockerfilePytorch
$ docker build -t aidk-pytorch110 . -f DockerfilePytorch110
```
