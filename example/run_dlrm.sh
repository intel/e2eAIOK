# prepare data
# tree ../dataset/criteo/
# ../dataset/criteo/
# ├── day_day_count.npz
# ├── day_fea_count.npz
# ├── metadata_dlrm_example.yaml
# ├── model_size.json
# ├── train
# │   └── train_data.bin
# └── valid
#     └── test_data.bin

# Use e2eaiok API
SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --data_path "/home/vmagent/app/dataset/criteo" --model_name dlrm --conf conf/e2eaiok_defaults_dlrm_example.conf

# Use SDA API
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --data_path "/home/vmagent/app/dataset/criteo" --model_name dlrm --conf conf/e2eaiok_defaults_dlrm_example.conf
