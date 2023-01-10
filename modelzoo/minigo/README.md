# Intel® End-to-End AI Optimization Kit for MiniGO
## Original source disclose
* Since MiniGo is a reinforcement learning model, it generates training dataset in each iteration during train loop and doesn't need dataset. We evaluate winrate with target model(based on MLPerf submission) and final winrate>=0.5, and all our optimizations guarantee that target metric.

* public reference on AlphaGo Zero: https://arxiv.org/abs/1712.01815

* public reference on MiniGo: https://openreview.net/forum?id=H1eerhIpLV

---

# Quick Start

## Data Processing
Get processed target model and bootstrap checkpoint (See the freeze target model section and convert selfplay data format section). Reference https://github.com/mlcommons/training_results_v1.0/tree/master/Intel/benchmarks/minigo/8-nodes-64s-8376H-tensorflow/ml_perf#steps-to-run-minigo


## Enviroment Setup
* Firstly, ensure that intel oneapi-hpckit and minigo conda runtime installed on server.
* Secondly, enter AIOK repo directory.
* Thirdly, start the jupyter notebook service.

``` bash
source /opt/intel/oneapi/setvars.sh --force
conda activate minigo
cd e2eAIOK
```

## Training
```
python run_e2eaiok.py --data_path /root/dataset/minigo --model_name minigo --conf conf/e2eaiok_defaults_minigo_example.conf
```
