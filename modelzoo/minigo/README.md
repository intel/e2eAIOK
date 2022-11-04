# Intel® End-to-End AI Optimization Kit for MiniGO
## Original source disclose
* Since MiniGo is a reinforcement learning model, it generates training dataset in each iteration during train loop and doesn't need dataset. We evaluate winrate with target model(based on MLPerf submission) and final winrate>=0.5, and all our optimizations guarantee that target metric.

* public reference on AlphaGo Zero: https://arxiv.org/abs/1712.01815

* public reference on MiniGo: https://openreview.net/forum?id=H1eerhIpLV

---

# Quick Start
## Enviroment Setup
* Firstly, ensure that intel oneapi-hpckit is installed on server.
* Secondly, enter AIOK repo directory.
* Thirdly, start the jupyter notebook service.

``` bash
source /opt/intel/oneapi/setvars.sh --force
conda activate minigo_xeon_opt
```

## Training
```
cd e2eAIOK; source /opt/intel/oneapi/setvars.sh --force && python run_e2eaiok.py --data_path /root/zheng/dataset/minigo --model_name minigo --conf conf/e2eaiok_defaults_minigo_example.conf
```