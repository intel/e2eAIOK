## Environment Setup

1. build docker image
   ```
   cd Dockerfile-ubuntu18.04 && docker build -t e2eaiok-pytorch120 . -f DockerfilePytorch120 && cd .. && yes | docker container prune && yes | docker image prune
   ```
2. run docker
   ```
   docker run -it --name UDA --privileged --network host --shm-size 32g --device=/dev/dri -v /mnt/DP_disk1/yu:/home/vmagent/app/dataset -v /home/yu:/work -w /work e2eaiok-pytorch120 /bin/bash 
   ``` 
3. apply patch
   ```
   cd e2eAIOK/ModelAdapter/src/task/medical_segmentation/third_party/nnUNet/nnunet
   patch -p1 < ../../../kits19.patch
   cd e2eAIOK/ModelAdapter/src/task/medical_segmentation
   cp -r third_party/nnUNet/nnunet .
   ```
4. install the development library
   ```
   source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
   conda activate pytorch-1.12.0
   cd e2eAIOK/ModelAdapter/src/task/medical_segmentation/
   pip install -e .
   ```
5. Start the jupyter notebook service
   ```
   pip install jupyter
   jupyter notebook --notebook-dir=/work --ip=0.0.0.0 --port=8989 --allow-root
   ```
   Now you can visit Adapter demo in `http://${hostname}:${port}/`.