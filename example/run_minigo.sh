# prepare data
# tree ../dataset/minigo/
#├── checkpoints
#│   └── mlperf07
#├── meta.yaml
#├── target
#│   ├── target.data-00000-of-00001
#│   ├── target.index
#│   ├── target.meta
#│   └── target.minigo
#├── train
#└── valid


# Use e2eaiok API
SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --data_path "/root/dataset/minigo" --model_name minigo --conf conf/e2eaiok_defaults_minigo_example.conf

# Use SDA API
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --data_path "/root/dataset/minigo" --model_name minigo --conf conf/e2eaiok_defaults_minigo_example.conf