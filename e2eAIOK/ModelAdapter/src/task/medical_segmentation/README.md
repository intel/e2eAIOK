## Environment Setup

1. build docker image
   ```
   cd Dockerfile-ubuntu18.04 && docker build -t aidk-pytorch110 . -f DockerfilePytorch110 && cd .. && yes | docker container prune && yes | docker image prune
   ```
2. run docker
   ```
   docker run -it --name UDA --privileged --network host --shm-size 32g --device=/dev/dri -v /mnt/DP_disk1/yu:/home/vmagent/app/dataset -v /home/yu:/work -w /work aidk-pytorch110 /bin/bash 
   ``` 
3. apply patch
   ```
   cd AIDK/TransferLearningKit/src/task/medical_segmentation/third_party/nnUNet/nnunet
   patch -p1 < ../../../kits19.patch
   cd AIDK/TransferLearningKit/src/task/medical_segmentation
   cp -r third_party/nnUNet/nnunet .
   ```
4. install the development library
   ```
   source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
   conda activate pytorch-1.10.0
   cd AIDK/TransferLearningKit/src/task/medical_segmentation/
   pip install -e .
   ```
5. Start the jupyter notebook service
   ```
   pip install jupyter
   jupyter notebook --notebook-dir=/work --ip=0.0.0.0 --port=8989 --allow-root
   ```
   Now you can visit Adapter demo in `http://${hostname}:${port}/`.