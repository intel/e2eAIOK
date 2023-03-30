# Intel® End-to-End AI Optimization Kit for RNN-T


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/demo/builtin/rnnt/RNNT_DEMO.ipynb)&emsp;&emsp;  <img width="20" height="20" src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"> [view DEMO source codes](https://github.com/intel/e2eAIOK/blob/main/demo/builtin/rnnt/RNNT_DEMO.ipynb)
---

## Original source disclose
Notes: RNN-T training is based on LibriSpeech train-clean-100 and evaluated on dev-clean, we evaluated WER with stock model (based on MLPerf submission) at train-clean-100 dataset, and final WER is 0.25, all the following optimization guarantee 0.25 WER. MLPerf submission took 38.7min with 8x A100 on LibriSpeech train-960h dataset.

public reference on train-clean-100: https://arxiv.org/pdf/1807.10893.pdf, https://arxiv.org/pdf/1811.00787.pdf

---

# Quick Start
## Enviroment Setup
``` bash
# Setup ENV
git clone https://github.com/intel/e2eAIOK.git
cd e2eAIOK
git submodule update --init --recursive
python3 scripts/start_e2eaiok_docker.py -b pytorch112 -w ${host0} ${host1} ${host2} ${host3} --proxy ""
```

## Enter Docker
```
sshpass -p docker ssh ${host0} -p 12345
```

## Workflow Prepare
``` bash
# prepare model codes
cd /home/vmagent/app/e2eaiok/modelzoo/rnnt/pytorch
bash patch_rnnt.sh

# Download Dataset
# Download and unzip dataset from https://www.openslr.org/12 to /home/vmagent/app/dataset/LibriSpeech

# Generate tokenizer and tokenize text
cd /home/vmagent/app/e2eaiok/modelzoo/rnnt/pytorch
bash scripts/preprocess_librispeech.sh
```

## Training
```
# edit /home/vmagent/app/e2eaiok/conf/e2eaiok_defaults_rnnt_example.conf
### GLOBAL SETTINGS ###
observation_budget: 1
save_path: /home/vmagent/app/e2eaiok/result/
ppn: 2
train_batch_size: 8
eval_batch_size: 8
iface: lo
hosts:
- localhost
epochs: 2
```

```
cd /home/vmagent/app/e2eaiok && python run_e2eaiok.py --data_path /home/vmagent/app/dataset/LibriSpeech --model_name rnnt --conf conf/e2eaiok_defaults_rnnt_example.conf 
```
