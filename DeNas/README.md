# Quick Start

```
scripts/run_docker
conda activate pytorch_1.10
pip install torchvision
```

# Run quick try

```
python denas.py --log=DEBUG --max_search_iter 2 --conf ../conf/hydroai_defaults_denas_cv.conf


{'domain': 'cv', 'max_search_iter': 2, 'budget_model_size': None, 'budget_flops': None, 'budget_latency': None, 'batch_size': 4096, 'save_dir': None, 'conf': '../conf/hydroai_defaults_denas_cv.conf', 'log': 'DEBUG', 'population_size': 512, 'num_classes': 1000, 'max_layers': 14, 'input_image_size': 224, 'no_reslink': True, 'no_BN': True, 'use_se': True, 'init_structure': 'SuperConvK3BNRELU(3,8,1,1)SuperResK3K3(8,16,1,8,1)SuperResK3K3(16,32,2,16,1)SuperResK3K3(32,64,2,32,1)SuperResK3K3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1)'}
---debug use_se in SuperResK3K3(8,16,1,8,1)
---debug use_se in SuperResK3K3(16,32,2,16,1)
---debug use_se in SuperResK3K3(32,64,2,32,1)
---debug use_se in SuperResK3K3(64,64,2,32,1)
DEBUG:root:num layers is 8
DEBUG:root:  Get score of random structure: SuperConvK3BNRELU(3,8,1,1)SuperResK3K3(8,16,1,8,1)SuperResK7K7(16,16,2,8,3)SuperResK3K3(16,64,2,32,1)SuperResK3K3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1) took 33.03823735000333 sec
DEBUG:root:score is 28.56313705444336
DEBUG:root:num layers is 8
DEBUG:root:  Get score of random structure: SuperConvK3BNRELU(3,8,1,1)SuperResK1K7K1(8,40,1,16,3)SuperResK3K3(40,32,2,16,1)SuperResK3K3(32,64,2,32,1)SuperResK3K3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1) took 186.0731563839945 sec
DEBUG:root:score is 38.043975830078125
DEBUG:root:Search for Best structure, took:  took 219.1387570130173 sec
DeNas search completed, best structure is [(38.043975830078125, inf, 'SuperConvK3BNRELU(3,8,1,1)SuperResK1K7K1(8,40,1,16,3)SuperResK3K3(40,32,2,16,1)SuperResK3K3(32,64,2,32,1)SuperResK3K3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1)')]
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

* Modify denas.py:94 to your implemeted scoring package function
```
nas_core_info = compute_zen_score.compute_nas_score(model=the_model, gpu=None,
                                                                        resolution=self.input_image_size,
                                                                        mixup_gamma=1e-2, batch_size=self.batch_size,
                                                                        repeat=1)
```

### define search strategy

* Modify denas.py:267 to your search strategy
```
searcher = DeNasEASearchEngine(main_net = DeMainNet, search_space = DeSearchSpace, settings = settings)
```
