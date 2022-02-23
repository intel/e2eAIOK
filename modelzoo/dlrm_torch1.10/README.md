# Notes
This ls DLRM for pytorch v1.10, and is under development. You can use docker `xuechendi/oneapi-aikit:hydro.ai` for pytorch v1.10 to evaluate the code.

This code has performance issue over `modelzoo/dlrm`, if you want to evaluate performance, please use model under `modelzoo/dlrm`, the corresponding docker is `xuechendi/oneapi-aikit:legacy_hydro.ai`

# Launch training with SDA
`python run_hydroai.py --data_path "/home/vmagent/app/dataset/criteo" --model_name dlrm_torch110 --conf conf/hydroai_defaults_dlrm_example.conf`