# Step by step guide on deploy BERT demo in AIDK docker

* Jupyter server: vsr257
* Port: 8888
* Dataset location:  /mnt/DP_disk2/TY/dataset/SQuAD
* Code location: /mnt/DP_disk1/tianyi/github/bert-demo/frameworks.bigdata.AIDK/modelzoo/bert

## Prepare work

* Copy the dataset and code to {dataset_path} and {aidk_code_path}
* Build the docker image

```
cd Dockerfile-ubuntu18.04
docker build -t e2eaiok-tensorflow . -f DockerfileTensorflow --build-arg http_proxy --build-arg https_proxy
```

* Create the docker container with e2eaiok-tensorflow

```
docker run -itd --name aidk-bert --privileged --network host --device=/dev/dri -v ${dataset_path}:/home/vmagent/app/dataset -v ${aidk_code_path}:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ e2eaiok-tensorflow /bin/bash
```

* Enter container

```
docker exec -it aidk-bert bash
```

* Start the jupyter notebook service

```
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
conda activate tensorflow-2.5.0
pip install jupyter
nohup jupyter notebook --notebook-dir=/home/vmagent/app/e2eaiok/ --ip=0.0.0.0 --port=8888 --allow-root &
```

Now you can visit AIDK BERT demo in http://vsr257:8888/notebooks/modelzoo/bert/demo/AIDK_BERT_DEMO.ipynb
