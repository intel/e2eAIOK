# Quick Start

# Environment Setup
``` bash
# Setup ENV
git clone https://github.com/intel/e2eAIOK.git
cd e2eAIOK
git submodule update --init --recursive
python3 scripts/start_e2eaiok_docker.py -b pytorch120 -w ${host0} ${host1} ${host2} ${host3} --proxy ""
```

## Enter Docker
```
sshpass -p docker ssh ${host0} -p 12347
```

# Enter DeNas directory
cd /home/vmagent/app/e2eaiok/e2eAIOK/DeNas

# Run quick try for CNN model

```
python -u search.py --domain cnn --conf ../../conf/denas/cv/e2eaiok_denas_cnn.conf
```

# Run quick try for ViT model

```
python -u search.py --domain vit --conf ../../conf/denas/cv/e2eaiok_denas_vit.conf
```

# Run quick try for Bert model

```
python -u search.py --domain bert --conf ../../conf/denas/nlp/e2eaiok_denas_bert.conf
```

# Run quick try for ASR model

```
python -u search.py --domain asr --conf ../../conf/denas/asr/e2eaiok_denas_asr.conf
```