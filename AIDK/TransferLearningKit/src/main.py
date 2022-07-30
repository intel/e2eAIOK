#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import time
from dataset.image_list import ImageList
from dataset.composed_dataset import ComposedDataset
from engine_core.backbone.factory import createBackbone
from engine_core.adapter.factory import createAdapter
from engine_core.transferrable_model import make_transferrable,TransferStrategy,TransferrableModel
from training.train import Trainer
import torch.optim as optim
from training.utils import EarlyStopping
from training.metrics import accuracy
import logging
from torchvision import transforms
def createDatasets():
    ''' create all datasets

    :return: (train_dataset, validation_dataset,test_dataset)
    '''
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        # transforms.Normalize(
        # mean=[0.485, 0.456, 0.406],
        # std=[0.229, 0.224, 0.225])
    ])
    target_test_dataset = ImageList("../datasets/USPS_vs_MNIST/MNIST",
                                    open("../datasets/USPS_vs_MNIST/MNIST/mnist_test.txt").readlines(),
                                    transform,'L')
    target_train_dataset = ImageList("../datasets/USPS_vs_MNIST/MNIST",
                                    open("../datasets/USPS_vs_MNIST/MNIST/mnist_train.txt").readlines(),
                                     transform,'L')
    source_train_dataset = ImageList("../datasets/USPS_vs_MNIST/USPS",
                                    open("../datasets/USPS_vs_MNIST/USPS/usps_train.txt").readlines(),
                                     transform,'L')

    test_len = len(target_test_dataset)
    target_test_dataset,target_validation_dataset = random_split(target_test_dataset,[test_len//2,test_len-test_len//2])
    return ComposedDataset(target_train_dataset,source_train_dataset),target_validation_dataset,target_test_dataset


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
    (train_dataset,validation_dataset,test_dataset) = createDatasets()
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
    epoch_steps = len(train_dataset)//batch_size
    logging.info("epoch_steps:%s"%epoch_steps)
    # adapter = None # createAdapter('DANN',input_size=model.adapter_size,hidden_size=500,dropout=0.0,
    #                         grl_coeff_alpha=5.0,grl_coeff_high=1.0,
    #                         max_iter=epoch_steps)
    adapter = createAdapter('CDAN',input_size=model.adapter_size*num_classes,hidden_size=model.adapter_size,
                            dropout=0.0,grl_coeff_alpha=5.0,grl_coeff_high=1.0,max_iter=epoch_steps,
                            backbone_output_size=num_classes,enable_random_layer=0,enable_entropy_weight=0)
    distiller = None
    logging.info('tensorboard_writer :%s' % tensorboard_writer)
    logging.info('early_stopping :%s' % early_stopping)
    logging.info('backbone:%s' % model)
    logging.info('adapter:%s' % adapter)
    logging.info('distiller:%s' % distiller)

    model = make_transferrable(model,adapter,distiller,TransferStrategy.OnlyDomainAdaptionStrategy,
                               enable_target_training_label=False)
    if (not isinstance(model,TransferrableModel)) and (isinstance(train_dataset,ComposedDataset)):
        raise RuntimeError("ComposedDataset can not be used in original model")
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    logging.info('transferrable model:%s' % model)
    #################################### train and evaluate ###################
    validate_metric_fn_map = {'acc':accuracy}
    trainer = Trainer(model,optimizer,early_stopping,validate_metric_fn_map,'acc', training_epochs,
                 tensorboard_writer,logging_interval_step)
    logging.info("trainer:%s"%trainer)
    trainer.train(train_loader,epoch_steps,validate_loader,"../model/model.pth")