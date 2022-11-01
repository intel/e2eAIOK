# Step by step guide on deploy MiniGo demo

* Jupyter server: sr141:8888
* Dataset Path: /root/zheng/dataset
* Code Path: /root/zheng/frameworks.bigdata.AIDK

## Prepare work

* Copy the dataset and code to {Dataset Path} and {Code Path}

* Create conda environment named `minigo_xeon_opt` by following guide in `modelzoo/minigo/ml_perf/README.md`

* Start the jupyter notebook service

```
source /opt/intel/oneapi/setvars.sh --force
conda activate minigo_xeon_opt
pip install jupyterlab
jupyter notebook --notebook-dir=./ --ip=0.0.0.0 --port=8888 --allow-root
```

Now you can visit AIDK MiniGo demo in http://sr141:8888/notebooks/demo/AIDK_MiniGo_DEMO.ipynb