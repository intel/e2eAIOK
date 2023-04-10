# clone repo
git clone https://github.com/intel/e2eAIOK.git
cd e2eAIOK
git submodule update --init -recursive

# apply patch
cd modelzoo/bert && bash patch_bert.sh

# download pre-trained model
mkdir -p /home/vmagent/app/dataset/SQuAD/pre-trained-model/bert-large-uncased/
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip -O /home/vmagent/app/dataset/SQuAD/pre-trained-model/bert-large-uncased/wwm_uncased_L-24_H-1024_A-16.zip
cd /home/vmagent/app/dataset/SQuAD/pre-trained-model/bert-large-uncased/ && unzip wwm_uncased_L-24_H-1024_A-16.zip

# download dataset
mkdir -p /home/vmagent/app/dataset/SQuAD
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O /home/vmagent/app/dataset/SQuAD/train-v1.1.json && wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O /home/vmagent/app/dataset/SQuAD/dev-v1.1.json