# Quick Start

### Build DeNas docker image


```
$ cd Dockerfile-ubuntu18.04
$ docker build -t aidk-denas-pytorch110 . -f DockerfilePytorch110
```

### Run DeNas docker container

```
$ docker run --shm-size=10g -it --privileged --network host --device=/dev/dri -v ${dataset_path}:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ aidk-denas-pytorch110 /bin/bash
$ conda activate pytorch-1.10.0
```

# Run quick try for CNN model

```
python -u search.py --domain cnn --conf ../conf/denas/cv/aidk_denas_cnn.conf
```

# Run quick try for ViT model

```
python -u search.py --domain vit --conf ../conf/denas/cv/aidk_denas_vit.conf
```

# Run quick try for Bert model

```
python -u search.py --domain bert --conf ../conf/denas/nlp/aidk_denas_bert.conf
```

# Run quick try for ASR model

```
python -u search.py --domain asr --conf ../conf/denas/asr/aidk_denas_asr.conf
```