# Quick Start

```
scripts/run_docker
conda activate pytorch_1.10
pip install torchvision
```

# Run quick try for CNN model

```
python denas.py --max_search_iter 1004 --conf ../conf/denas/aidk_denas_cnn.conf


{'domain': 'cv', 'max_search_iter': 1002, 'budget_model_size': 1000000, 'budget_flops': 10000000, 'budget_latency': 1, 'batch_size': 64, 'save_dir': '/Zen_NAS_search', 'conf': '../conf/aidk_denas_cv.conf', 'log': 'INFO', 'zero_shot_score': 'De_score2', 'search_space': 'SearchSpace/search_space_XXBL.py', 'max_layers': 18, 'input_image_size': 32, 'plainnet_struct_txt': 'SuperConvK3BNRELU(3,8,1,1)SuperResK3K3(8,16,1,8,1)SuperResK3K3(16,32,2,16,1)SuperResK3K3(32,64,2,32,1)SuperResK3K3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1)', 'num_classes': 100, 'evolution_max_iter': 1004, 'population_size': 10, 'no_reslink': False, 'no_BN': False, 'use_se': False}
loop_count=1000/1002, max_score=0.130189, min_score=0.000433214, time=0.0466935h
DeNas search completed, best structure is [(tensor(0.1302), 0.000433214008808136, 'SuperConvK3BNRELU(3,8,1,1)SuperResK3K3(8,16,1,8,1)SuperResK3K3(16,32,2,16,1)SuperResK3K3(32,64,2,32,1)SuperResK1K3K1(64,24,2,64,3)SuperConvK1BNRELU(24,128,1,1)')]
DeNasSearchEngine destructed.
```



# Advanced

### define mainnet structure

* Modify denas.py:260 to your main net and search space
```
if settings["domain"] == "cv":
        from cv.third_party.ZenNet import DeSearchSpaceXXBL as DeSearchSpace
        from cv.third_party.ZenNet import DeMainNet as DeMainNet
```

* implement main net with below API
```
class DeSearchSpace:
    def __init__():
        pass

class DeMainNet:
    def __init__():
        pass

    @classmethod
    def create_netblock_list_from_str(cls, struct_str, no_create=False, **kwargs):
        pass
```

### define scoring system

* Modify scores/compute_de_score.py:228 to your implemeted scoring package function
```
def do_compute_nas_score(model_type, model, resolution, batch_size, mixup_gamma):
    if model_type == "cnn":
        do_compute_nas_score_cnn(model, resolution, batch_size, mixup_gamma)
    elif model_type == "transformer":
        do_compute_nas_score_transformer(model, resolution, batch_size, mixup_gamma)
```

### define search strategy

* Modify denas.py:267 to your search strategy
```
searcher = DeNasEASearchEngine(main_net = DeMainNet, search_space = DeSearchSpace, settings = settings)
```
