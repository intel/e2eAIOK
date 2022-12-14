# prepare data
# tree -d 1 ../dataset/outbrain/
# 1 [error opening dir]
# ../dataset/outbrain/
# ├── meta
# │   └── transformed_metadata
# ├── train
# └── valid

# change hosts and eth in conf/e2eaiok_defaults_wnd_example.conf
# iface: ${eth0}
# hosts:
#   - ${host_name}


# Use e2eaiok API
SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --data_path "/home/vmagent/app/dataset/outbrain" --model_name wnd --conf conf/e2eaiok_defaults_wnd_example.conf

# Use SDA API
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --data_path "/home/vmagent/app/dataset/outbrain" --model_name wnd --conf conf/e2eaiok_defaults_wnd_example.conf
