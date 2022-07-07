# Quick Start

### Build DeNas docker image

* Before building docker image, firstly create SSH private key `id_rsa` under folder `Dockerfile-ubuntu18.04` to enbale passwordless SSH.

```
$ cd Dockerfile-ubuntu18.04
$ docker build -t aidk-denas-pytorch110 . -f DockerfilePytorch110
```

### Run DeNas docker container

```
$ docker run --shm-size=10g -it --privileged --network host --device=/dev/dri -v ${dataset_path}:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ aidk-denas-pytorch110 /bin/bash
$ conda activate pytorch_1.10
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

# Advanced

### define scoring system

* Modify scores/compute_de_score.py:100 to your implemeted scoring package function

```
def do_compute_nas_score(model_type, model, resolution, batch_size, mixup_gamma):
    if model_type == "cnn":
        do_compute_nas_score_cnn(model, resolution, batch_size, mixup_gamma)
    elif model_type == "transformer":
        do_compute_nas_score_transformer(model, resolution, batch_size, mixup_gamma)
```
