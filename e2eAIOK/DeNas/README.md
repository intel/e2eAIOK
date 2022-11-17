# Quick Start

### Build DeNas docker image


```
$ cd Dockerfile-ubuntu18.04
$ docker build -t e2eaiok-pytorch110 . -f DockerfilePytorch110
```

### Run DeNas docker container

```
$ docker run --shm-size=10g -it --privileged --network host --device=/dev/dri -v ${dataset_path}:/home/vmagent/app/dataset -v ${e2eaiok_codebase}:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ e2eaiok-pytorch110 /bin/bash
$ conda activate pytorch-1.10.0
```

# Run quick try for CNN model

```
python -u search.py --domain cnn --conf ../../conf/denas/cv/e2eaiok_denas_cnn.conf
```

# Run quick try for ViT model

```
python -u search.py --domain vit --conf ../../conf/denas/cv/e2eaiok_denas_vit.conf
```

# Run quick try for Bert model

```
python -u search.py --domain bert --conf ../../conf/denas/nlp/e2eaiok_denas_bert.conf
```

# Run quick try for ASR model

```
python -u search.py --domain asr --conf ../../conf/denas/asr/e2eaiok_denas_asr.conf
```