# Step by step guide on deploy RNN-T demo in AIDK docker

* Jupyter server: vsr602
* Dataset location: /root/ht/LibriSpeech-cicd
* Code location: /root/ht/rnnt-demo/frameworks.bigdata.bluewhale

## Prework

apply RNN-T patch
```bash
cd /root/ht/rnnt-demo/frameworks.bigdata.bluewhale/modelzoo/rnnt/pytorch
bash patch_rnnt.sh
```
set `${EPOCH:=5}` and `--nnodes=1` in scripts/train.sh, set `BATCHSIZE=16` and `EVAL_BATCHSIZE=16` in config.sh

set `${TRAIN_MANIFESTS:="$META_DIR/train-test.json"}` and `${VAL_MANIFESTS:="$META_DIR/dev-test.json"}` in scripts/train.sh

```bash
# start docker
docker run -itd --name aidk-rnnt --privileged --network host --device=/dev/dri -v /root/ht/LibriSpeech-cicd/:/home/vmagent/app/dataset/LibriSpeech -v /root/ht/rnnt-demo/frameworks.bigdata.bluewhale/:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ e2eaiok-pytorch110:latest /bin/bash
docker exec -it aidk-rnnt bash

source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
conda activate pytorch-1.10.0
pip install jupyter
jupyter notebook password
nohup jupyter notebook --notebook-dir=/home/vmagent/app/e2eaiok --ip=0.0.0.0 --port=8888 --allow-root &
```

Now the demo notebook can be accessed through http://sr602:8888