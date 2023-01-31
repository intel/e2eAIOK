# NLP Workflow

## Geting Started

### Environment Setup

```bash
# Setup ENV
git clone https://github.com/intel/e2eAIOK.git
cd e2eAIOK
git submodule update --init --recursive
python3 scripts/start_e2eaiok_docker.py -b pytorch112 -w ${host0} ${host1} ${host2} ${host3} --proxy ""
```

 Enter Docker

```
sshpass -p docker ssh ${host0} -p 12347
```

Enter DeNas directory

```
cd /home/vmagent/app/e2eaiok/e2eAIOK/DeNas
```

#### Trial for DE-NAS BERT model

Running Search

```
python -u search.py --domain bert --conf ../../conf/denas/nlp/e2eaiok_denas_bert.conf
```

Running Training

```
python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=1 --nnodes=1 train.py --domain bert --conf ../../conf/denas/nlp/e2eaiok_denas_train_bert.conf
```

#### Trial for DE-NAS and Distiller on BERT model

Running Search

```
python -u search.py --domain bert --conf ../../conf/denas/nlp/e2eaiok_denas_bert.conf
```

Running Training

*Phrase1: Prepare Logits*

```
# Set the is_saving_logits to be True for triggering the logits preparation process
sed -i '/is_saving_logits:/ s/:.*/: True/' ../../conf/denas/nlp/e2eaiok_workflow_train_bert.conf
python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=1 --nnodes=1 train.py --domain bert --conf ../conf/denas/nlp/e2eaiok_workflow_train_bert.conf
```

*Phrase2: Trigger the DE-NAS trainer with KD*

```
# Set the is_saving_logits to be False for triggering the DE-NAS training with KD process
sed -i '/is_saving_logits:/ s/:.*/: False/' ../../conf/denas/nlp/e2eaiok_workflow_train_bert.conf
python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=1 --nnodes=1 train.py --domain bert --conf ../conf/denas/nlp/e2eaiok_workflow_train_bert.conf
```
