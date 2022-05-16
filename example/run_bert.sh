# prepare data
# tree ../dataset/SQuAD/
# ├── dev-v1.1.json
# ├── evaluate-v1.1.py
# ├── squad.meta.yaml
# ├── test-v1.1.json
# ├── train
# │   └── train-v1.1.json
# ├── train-v1.1.json
# └── valid
#     └── dev-v1.1.json

# Use hydro.ai API
SIGOPT_API_TOKEN=${TOKEN} python run_hydroai.py --data_path "/home/vmagent/app/dataset/SQuAD" --model_name bert --conf conf/hydroai_defaults_bert_example.conf

# Use SDA API
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --data_path "/home/vmagent/app/dataset/SQuAD" --model_name bert --conf conf/hydroai_defaults_bert_example.conf
