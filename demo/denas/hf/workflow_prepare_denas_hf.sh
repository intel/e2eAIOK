# download pre-trained model
git lfs install
cd /home/vmagent/app/dataset/ && git clone https://huggingface.co/bert-base-uncased

# download dataset
mkdir -p /home/vmagent/app/dataset/SQuAD
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O /home/vmagent/app/dataset/SQuAD/train-v1.1.json && wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O /home/vmagent/app/dataset/SQuAD/dev-v1.1.json