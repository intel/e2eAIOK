# Step by step guide on deploy DE-NAS and TLK on BERT with Hugging Face demo in AIDK docker

* Jupyter server: vsr257
* Port: 8890
* Dataset localtion: /mnt/DP_disk2/TY/dataset/SQuAD/
* Code location: /mnt/DP_disk2/TY/AIDK_DENAS_DEMO/

## Prework

* Copy the dataset and code to {dataset_path} and {aidk_code_path}
* Build the docker image:

```
cd Dockerfile-ubuntu18.04
docker build -t aidk-pytorch110 . -f DockerfilePytorch110 --build-arg http_proxy --build-arg https_proxy
docker run -itd --name aidk-denas-bert --privileged --network host --device=/dev/dri -v ${dataset_path}:/home/vmagent/app/dataset -v ${aidk_code_path}:/home/vmagent/app/aidk -w /home/vmagent/app/ aidk-pytorch110 /bin/bash

```

* Enter container with `docker exec -it aidk-denas-bert bash`
* Install the jupyter and Hugging Face API

```
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
conda activate pytorch-1.10.0
pip install jupyter
pip install transformers[torch]
```

* Start the jupyter notebook service

`nohup jupyter notebook --notebook-dir=/home/vmagent/app/aidk/ --ip=0.0.0.0 --port=8890 --allow-root &`

* Now you can visit DE-NAS BERT demo in http://vsr257:8888/notebooks/DeNas/demo/DENAS-BERT-TLK-DEMO.ipynb
