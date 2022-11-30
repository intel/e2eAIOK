# Step by step guide on deploy RNN-T demo in AIDK docker

* Dataset location: sr113:/mnt/nvm2/LibriSpeech-speechbrain
* Code location: /root/tmp/frameworks.bigdata.AIDK
* Dependency: /root/tmp/frameworks.bigdata.AIDK/DeNas/asr/results

## Prework

copy dependency results folder to `frameworks.bigdata.AIDK/DeNas/asr`

## Start jupyter server

```bash
# start docker
docker run -itd --name aidk-denas-asr --privileged --network host --device=/dev/dri -v /mnt/nvm2/LibriSpeech-speechbrain/:/home/vmagent/app/dataset/LibriSpeech -v ${AIDK_code_path}:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ e2eaiok-pytorch:latest /bin/bash
docker exec -it aidk-denas-asr bash

source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
conda activate pytorch-1.10.0
pip install jupyter
jupyter notebook password
nohup jupyter notebook --notebook-dir=/home/vmagent/app/e2eaiok --ip=0.0.0.0 --port=8888 --allow-root &
```

Now the demo notebook can be accessed through http://${hostname}:8888