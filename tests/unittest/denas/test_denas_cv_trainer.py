from easydict import EasyDict as edict
from e2eAIOK.DeNas.cv.cv_trainer import CVTrainer
import e2eAIOK.common.trainer.utils.utils as utils
from e2eAIOK.DeNas.cv.model_builder_denas_cv import ModelBuilderCVDeNas
from e2eAIOK.common.trainer.data.cv.data_builder_cifar import DataBuilderCIFAR

class TestDeNasCVTrainer:

    '''
    Test Unified API DE-NAS CNN train
    '''
    def test_cnn_trainer(self):
        cnn_structure = "SuperConvK3BNRELU(3,8,1,1)SuperResK3K3(8,16,1,8,1)SuperResK3K3(16,32,2,16,1)SuperResK3K3(32,64,2,32,1)SuperResK3K3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1)"
        with open("./best_model_structure.txt", 'w') as f:
            f.write(str(cnn_structure))
        cfg = edict({'domain': 'cnn', 'train_epochs': 1, 'eval_epochs': 1, 'input_size': 32,
              'best_model_structure': './best_model_structure.txt', 'num_classes': 10, 
              'dist_backend': 'gloo', 'train_batch_size': 128, 'eval_batch_size': 128, 
              'data_path': '~/data/pytorch_cifar10', 'data_set': 'CIFAR10', 
              'output_dir': './', 'num_workers': 10, 'pin_mem': True, 'eval_metric': 'accuracy', 
              'learning_rate': 0.001, 'momentum': 0.9, 'weight_decay': 0.01, 'optimizer': 'SGD',
               'criterion': 'CrossEntropyLoss', 'lr_scheduler': 'CosineAnnealingLR', 'print_freq': 10,
                'metric_threshold': 94, 'mode': 'train'})
        model = ModelBuilderCVDeNas(cfg).create_model()
        train_dataloader, eval_dataloader = DataBuilderCIFAR(cfg).get_dataloader()
        optimizer = utils.create_optimizer(model, cfg)
        criterion = utils.create_criterion(cfg)
        scheduler = utils.create_scheduler(optimizer, cfg)
        metric = utils.create_metric(cfg)
        trainer = CVTrainer(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric)
        best_acc = 0
        best_acc = trainer.fit()
        
        assert best_acc > 0

    '''
    Test Unified API DE-NAS ViT train
    '''  
    def test_vit_trainer(self):
        vit_structure = "(13, 3.5, 3.5, 3.0, 3.0, 3.0, 4.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0,4.0, 3, 3, 3, 9, 4, 7, 10, 7, 3, 6, 3, 5, 10, 216)"
        with open("./best_model_structure.txt", 'w') as f:
            f.write(str(vit_structure))
        cfg = edict({'domain': 'vit', 'train_epochs': 1, 'eval_epochs': 1, 'input_size': 32, 
               'best_model_structure': './best_model_structure.txt', 'num_classes': 10,
               'dist_backend': 'gloo', 'train_batch_size': 128, 'eval_batch_size': 128, 
               'data_path': '~/data/pytorch_cifar10', 'data_set': 'CIFAR10', 'output_dir': './',
               'num_workers': 10, 'pin_mem': True, 'eval_metric': 'accuracy', 'learning_rate': 0.001,
               'momentum': 0.9, 'weight_decay': 0.01, 'optimizer': 'SGD', 'criterion': 'CrossEntropyLoss',
               'lr_scheduler': 'CosineAnnealingLR', 'print_freq': 10, 'mode': 'train', 'gp': True,
               'change_qkv': True, 'relative_position': True, 'drop_path': 0.1, 'max_relative_position': 14,
               'no_abs_pos': False, 'patch_size': 16, 'drop':0.0, 'metric_threshold': 94,
               'SUPERNET': {'MLP_RATIO': 4.0, 'NUM_HEADS': 10, 'EMBED_DIM': 640, 'DEPTH': 16}})
        model = ModelBuilderCVDeNas(cfg).create_model()
        train_dataloader, eval_dataloader = DataBuilderCIFAR(cfg).get_dataloader()
        optimizer = utils.create_optimizer(model, cfg)
        criterion = utils.create_criterion(cfg)
        scheduler = utils.create_scheduler(optimizer, cfg)
        metric = utils.create_metric(cfg)
        trainer = CVTrainer(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric)
        best_acc = 0
        best_acc = trainer.fit()
        
        assert best_acc > 0
