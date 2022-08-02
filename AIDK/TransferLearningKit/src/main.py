#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import time
from dataset.image_list import ImageList

from engine_core.backbone.factory import createBackbone
from engine_core.adapter.factory import createAdapter
from engine_core.transferrable_model import make_transferrable,TransferStrategy,TransferrableModel
from training.train import Trainer
import torch.optim as optim
from training.utils import EarlyStopping
from training.metrics import accuracy
import logging
from torchvision import transforms
import torch.nn as nn
if __name__ == '__main__':
    logging.basicConfig(filename="../log/%s.txt"%int(time.time()), level=logging.INFO,
                        format='%(asctime)s %(levelname)s [%(filename)s %(funcName)s %(lineno)d]: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filemode='w')

    torch.manual_seed(0)
    batch_size = 128
    num_workers = 1

    learning_rate = 0.01
    weight_decay = 0.0005
    momentum = 0.9

    training_epochs = 2
    logging_interval_step = 10

    num_classes = 10
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    ######################## create dataset and dataloader #####################
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        # transforms.Normalize(
        # mean=[0.485, 0.456, 0.406],
        # std=[0.229, 0.224, 0.225])
    ])
    test_dataset = ImageList("../datasets/USPS_vs_MNIST/MNIST",
                                    open("../datasets/USPS_vs_MNIST/MNIST/mnist_test.txt").readlines(),
                                    transform, 'L')
    train_dataset = ImageList("../datasets/USPS_vs_MNIST/MNIST",
                                     open("../datasets/USPS_vs_MNIST/MNIST/mnist_train.txt").readlines(),
                                     transform, 'L')
    test_len = len(test_dataset)
    test_dataset, validation_dataset = random_split(test_dataset,[test_len // 2, test_len - test_len // 2])

    logging.info("train_dataset:" + str(train_dataset))
    logging.info("validation_dataset:" + str(validation_dataset))
    logging.info("test_dataset:" + str(test_dataset))


    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)
    validate_loader = torch.utils.data.DataLoader(validation_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)
    # ########################## create other components ############
    tensorboard_writer = SummaryWriter("../tensorboard_log",filename_suffix="_adapter")
    early_stopping = EarlyStopping(tolerance_epoch=3, delta=0.0001, is_max=True)
    model = createBackbone('LeNet',num_classes=num_classes)
    loss = nn.CrossEntropyLoss()
    epoch_steps = len(train_dataset)//batch_size
    logging.info("epoch_steps:%s"%epoch_steps)
    # adapter = None # createAdapter('DANN',input_size=model.adapter_size,hidden_size=500,dropout=0.0,
    #                         grl_coeff_alpha=5.0,grl_coeff_high=1.0,
    #                         max_iter=epoch_steps)

    logging.info('tensorboard_writer :%s' % tensorboard_writer)
    logging.info('early_stopping :%s' % early_stopping)
    logging.info('backbone:%s' % model)

    distiller_feature_size = None
    distiller_feature_layer_name = 'x'
    distiller = None

    adapter_feature_size = 500
    adapter_feature_layer_name = 'fc_layers_2'
    adapter = createAdapter('CDAN', input_size=adapter_feature_size * num_classes, hidden_size=adapter_feature_size,
                            dropout=0.0, grl_coeff_alpha=5.0, grl_coeff_high=1.0, max_iter=epoch_steps,
                            backbone_output_size=num_classes, enable_random_layer=0, enable_entropy_weight=0)

    model = make_transferrable(model,loss,distiller_feature_size,distiller_feature_layer_name,
                               adapter_feature_size, adapter_feature_layer_name,
                               distiller,adapter,train_loader,ImageList("../datasets/USPS_vs_MNIST/USPS",
                                    open("../datasets/USPS_vs_MNIST/USPS/usps_train.txt").readlines(),
                                     transform,'L'),
                               TransferStrategy.OnlyDomainAdaptionStrategy,
                               enable_target_training_label=False)
    logging.info('adapter:%s' % adapter)
    logging.info('distiller:%s' % distiller)
    logging.info('transferrable model:%s' % model)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    #################################### train and evaluate ###################
    validate_metric_fn_map = {'acc':accuracy}
    trainer = Trainer(model,optimizer,early_stopping,validate_metric_fn_map,'acc', training_epochs,
                 tensorboard_writer,logging_interval_step)
    logging.info("trainer:%s"%trainer)
    trainer.train(train_loader,epoch_steps,validate_loader,"../model/model.pth")