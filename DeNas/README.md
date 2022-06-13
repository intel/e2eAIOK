# Quick Start

```
scripts/run_docker
conda activate pytorch_1.10
pip install torchvision
```

# Run quick try for CNN model

```
python denas.py --max_search_iter 1004 --conf ../conf/denas/cv/aidk_denas_cnn.conf
```
Below is the simplified log:

```

{'domain': 'cv', 'max_search_iter': 1002, 'budget_model_size': 1000000, 'budget_flops': 10000000, 'budget_latency': 1, 'batch_size': 64, 'save_dir': '/Zen_NAS_search', 'conf': '../conf/aidk_denas_cv.conf', 'log': 'INFO', 'zero_shot_score': 'De_score2', 'search_space': 'SearchSpace/search_space_XXBL.py', 'max_layers': 18, 'input_image_size': 32, 'plainnet_struct_txt': 'SuperConvK3BNRELU(3,8,1,1)SuperResK3K3(8,16,1,8,1)SuperResK3K3(16,32,2,16,1)SuperResK3K3(32,64,2,32,1)SuperResK3K3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1)', 'num_classes': 100, 'evolution_max_iter': 1004, 'population_size': 10, 'no_reslink': False, 'no_BN': False, 'use_se': False}
loop_count=1000/1002, max_score=0.130189, min_score=0.000433214, time=0.0466935h
DeNas search completed, best structure is [(tensor(0.1302), 0.000433214008808136, 'SuperConvK3BNRELU(3,8,1,1)SuperResK3K3(8,16,1,8,1)SuperResK3K3(16,32,2,16,1)SuperResK3K3(32,64,2,32,1)SuperResK1K3K1(64,24,2,64,3)SuperConvK1BNRELU(24,128,1,1)')]
DeNasSearchEngine destructed.
```

# Run quick try for ViT model

```
python -u evolution.py --gp --change_qk --relative_position --model_type "transformer" --dist-eval --cfg ../conf/denas/cv/supernet_vit/supernet_base.conf --data-set CIFAR
```
Below is the simplified log:

```

Namespace(aa='rand-m9-mstd0.5-inc1', amp=True, batch_size=64, cfg='../conf/denas/supernet/supernet-B.yaml', change_qkv=True, clip_grad=None, color_jitter=0.4, cooldown_epochs=10, crossover_num=25, cutmix=1.0, cutmix_minmax=None, data_path='/datasets01_101/imagenet_full_size/061417/', data_set='CIFAR', decay_epochs=30, decay_rate=0.1, device='cuda', dist_eval=True, dist_url='env://', distributed=False, drop=0.0, drop_block=None, drop_path=0.1, epochs=1, eval=False, gp=True, inat_category='name', input_size=224, lr=0.0005, lr_noise=None, lr_noise_pct=0.67, lr_noise_std=1.0, lr_power=1.0, m_prob=0.2, max_epochs=1, max_relative_position=14, min_lr=1e-05, min_param_limits=18, mixup=0.8, mixup_mode='batch', mixup_prob=1.0, mixup_switch_prob=0.5, model='', model_ema=False, model_ema_decay=0.99996, model_ema_force_cpu=False, model_type='transformer', momentum=0.9, mutation_num=25, no_abs_pos=False, no_prefetcher=False, num_workers=10, opt='adamw', opt_betas=None, opt_eps=1e-08, output_dir='', param_limits=100, patch_size=16, patience_epochs=10, pin_mem=True, platform='pai', population_num=50, post_norm=False, recount=1, relative_position=True, remode='pixel', repeated_aug=True, reprob=0.25, resplit=False, resume='', rpe_type='bias', s_prob=0.4, scale=False,sched='cosine', seed=0, select_num=10, smoothing=0.1, start_epoch=0, teacher_model='', train_interpolation='bicubic', warmup_epochs=5, warmup_lr=1e-06, weight_decay=0.05, world_size=1)
Creating SuperVisionTransformer
{'SUPERNET': {'MLP_RATIO': 4.0, 'NUM_HEADS': 10, 'EMBED_DIM': 640, 'DEPTH': 16}, 'SEARCH_SPACE': {'MLP_RATIO': [3.0, 3.5, 4.0], 'NUM_HEADS': [9, 10], 'DEPTH': [14, 15, 16], 'EMBED_DIM': [528, 576, 624]}}
number of params: 79525770
population_num = 50 select_num = 10 mutation_num = 25 crossover_num = 25 random_num = 0 max_epochs = 1
random select ........
info['params']:64.340602, 18
rank: 0 (15, 3.5, 3.0, 3.5, 4.0, 3.5, 3.5, 3.5, 3.5, 3.5, 4.0, 3.0, 4.0, 3.0, 3.5, 3.0, 9, 10, 9, 10, 9, 9, 10, 10, 9, 10, 10,10, 9, 10, 10, 624) 64.340602

random 1/50
info['params']:65.029858, 18
rank: 0 (15, 3.0, 4.0, 3.0, 3.0, 4.0, 3.5, 4.0, 4.0, 4.0, 3.0, 4.0, 3.5, 3.5, 3.0, 4.0, 10, 9, 9, 9, 9, 9, 10, 9, 9, 10, 10, 9, 10, 10, 9, 624) 65.029858
random 2/50
info['params']:64.709986, 18
rank: 0 (15, 4.0, 3.0, 4.0, 4.0, 4.0, 3.5, 3.5, 3.0, 4.0, 3.5, 3.5, 4.0, 3.0, 3.5, 3.0, 9, 9, 9, 10, 10, 9, 9, 9, 9, 9, 9, 10,10, 9, 9, 624) 64.709986
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
