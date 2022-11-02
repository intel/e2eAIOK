# Step by step guide on deploy RNN-T demo in AIDK docker

* Jupyter server: vsr602
* Dataset location: /root/ht/outbrain-cicd
* Code location: /root/ht/wnd-demo/e2eAIOK

## Prework

apply WnD patch
```
cd /root/ht/wnd-demo/e2eAIOK/modelzoo/WnD/TensorFlow2
bash patch_wnd.patch
```

```bash
# start docker
docker run -itd --name aidk-wnd --privileged --network host --device=/dev/dri -v /root/ht/outbrain-cicd/:/home/vmagent/app/dataset/outbrain -v /root/ht/wnd-demo/e2eAIOK/:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ e2eaiok-tensorflow:latest /bin/bash
docker exec -it aidk-wnd bash

source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
conda activate tensorflow
pip install jupyter
jupyter notebook password
nohup jupyter notebook --notebook-dir=/home/vmagent/app/e2eaiok --ip=0.0.0.0 --port=8886 --allow-root &
```

Now the demo notebook can be accessed through http://sr602:8886