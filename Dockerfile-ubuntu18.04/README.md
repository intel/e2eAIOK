## Build Dockerfile for AIDK workloads

Before building AIDK docker image, firstly create SSH private key (id_rsa) under this folder to enbale passwordless SSH.
```
$ cd Dockerfile-ubuntu18.04
$ docker build -t aidk-base . -f DockerfileBase
$ docker build -t aidk-tensorflow . -f DockerfileTensorflow
$ docker build -t aidk-pytorch . -f DockerfilePytorch
$ docker build -t aidk-pytorch110 . -f DockerfilePytorch110
```
