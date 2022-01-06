# dataset layout
# dataset
# ├── tpcxai.yaml
# ├── train
# │   ├── CUSTOMER_IMAGES
# │   ├── CUSTOMER_IMAGES_META.csv
# │   └── CUSTOMER_IMAGES.seq
# ├── uc9_res
# │   ├── nn4.small2.v1.h5
# │   └── shape_predictor_5_face_landmarks.dat
# └── valid
#     ├── CUSTOMER_IMAGES
#     ├── CUSTOMER_IMAGES_META.csv
#     ├── CUSTOMER_IMAGES_META_labels.csv
#     └── CUSTOMER_IMAGES.seq

# Use hydro.ai API
SIGOPT_API_TOKEN=${TOKEN} python /home/vmagent/app/hydro.ai/run_hydroai.py --data_path /home/vmagent/app/dataset/tpcxai --model_name tpcxai09 --conf /home/vmagent/app/hydro.ai/conf/TPCxAI/hydroai_defaults_tpcxai_uc9_example.conf

# Use SDA API
SIGOPT_API_TOKEN=${TOKEN} python /home/vmagent/app/hydro.ai/SDA/SDA.py --data_path /home/vmagent/app/dataset/tpcxai --model_name tpcxai09 --conf /home/vmagent/app/hydro.ai/conf/TPCxAI/hydroai_defaults_tpcxai_uc9_example.conf
