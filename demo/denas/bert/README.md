# Step by step guide on deploy BERT demo in AIDK docker

* Jupyter server: vsr262
* Port: 8888
* Dataset location: /mnt/DP_disk1/TY/dataset
* Code location: /mnt/DP_disk1/TY/github/denas_demo

## Prework

* Copy the dataset and code to '{dataset_path}' and '{aidk_code_path}'
* Build docker image

  ```
  cd Dockerfile-ubuntu18.04
  docker build -t aidk-pytorch110 . -f DockerfilePytorch110 --build-arg http_proxy --build-arg https_proxy
  docker run -itd --name aidk-denas-bert --privileged --network host --device=/dev/dri -v ${dataset_path}:/home/vmagent/app/dataset -v ${aidk_code_path}:/home/vmagent/app/aidk -w /home/vmagent/app/ aidk-pytorch110 /bin/bash
  ```
* Enter container

  ```
  docker exec -it aidk-denas-bert bash
  ```
* Install the jupyter

  ```
  source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
  conda activate pytorch-1.10.0
  pip install jupyter
  ```
* Start the jupyter notebook service

  ```
  nohup jupyter notebook --notebook-dir=/home/vmagent/app/aidk/ --ip=0.0.0.0 --port=8888 --allow-root &
  ```

Now you can visit DE-NAS BERT demo in http://vsr262:8888/notebooks/DeNas/demo/denas/bert/AIDK_DENAS_BERT_DEMO.ipynb
