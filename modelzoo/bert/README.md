# Intel® End-to-End AI Optimization Kit for BERT

## Original source disclose

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. SQuAD 1.1 contains 100,000+ question-answer pairs on 500+ articles.

original code source: https://github.com/IntelAI/models.git

---

# Quick Start

## Enviroment Setup

```bash
# Setup ENV
git clone https://github.com/intel/e2eAIOK.git
cd e2eAIOK
git submodule update --init --recursive
python3 scripts/start_e2eaiok_docker.py -b tensorflow -w ${host0} ${host1} ${host2} ${host3} --proxy ""
```
# Download Dataset
* Download from below path to /home/vmagent/app/dataset/SQuAD

    * Train Data: [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
    * Test Data: [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
    * Data Format:

{
    "answers": {
        "answer_start": [1],
        "text": ["This is a test text"]
    },
    "context": "This is a test context.",
    "id": "1",
    "question": "Is this a test?",
    "title": "train test"
}
* Download Pre-trained models to /home/vmagent/app/dataset/SQuAD/pre-trained-model/bert-large-uncased/
Download and extract one of BERT large pretrained models from [Google BERT repository](https://github.com/google-research/bert#pre-trained-models) 

## Enter Docker

```
sshpass -p docker ssh ${host0} -p 12344
```

## Workflow Prepare

```bash
# prepare model codes
cd /home/vmagent/app/e2eaiok/modelzoo/bert
sh patch_bert.sh
```

## Training

```
cd /home/vmagent/app/e2eaiok/; python run_e2eaiok.py --data_path /home/vmagent/app/dataset/SQuAD --model_name bert --conf /home/vmagent/app/e2eaiok/conf/e2eaiok_defaults_bert_example.conf 
```
