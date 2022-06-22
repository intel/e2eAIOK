# Quick Start

```
scripts/run_docker
conda activate pytorch_1.10
pip install torchvision torchsummary easydict opencv-python scikit-image
```

# Run quick try for CNN model

```
python -u search.py --domain cnn --conf ../conf/denas/cv/aidk_denas_cnn.conf
```

# Run quick try for ViT model

```
python -u search.py --domain vit --conf ../conf/denas/cv/aidk_denas_vit.conf
```

# Advanced

### define scoring system

* Modify scores/compute_de_score.py:100 to your implemeted scoring package function
```
def do_compute_nas_score(model_type, model, resolution, batch_size, mixup_gamma):
    if model_type == "cnn":
        do_compute_nas_score_cnn(model, resolution, batch_size, mixup_gamma)
    elif model_type == "transformer":
        do_compute_nas_score_transformer(model, resolution, batch_size, mixup_gamma)
```
