import pytest
import sys
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from e2eAIOK.ModelAdapter.src.engine_core.distiller import BasicDistiller, KD, DKD
from e2eAIOK.ModelAdapter.src.engine_core.distiller.utils import logits_wrap_dataset
from torch.utils.data import Subset
import random

torch.manual_seed(0)
random.seed(32)

class TestBasicDistiller:
    ''' Test BasicDistiller
    
    '''
    def _create_dataloader(self, save_logits):
        trans = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
        
        dataset = datasets.CIFAR100(root="/home/vmagent/app/data/dataset", train=False, transform=trans, download=True)
        dataset = Subset(dataset, torch.Tensor([i for i in range(200)]).int())
        dataset = logits_wrap_dataset(dataset, logits_path="/home/vmagent/app/data/dataset", num_classes=100, save_logits=save_logits, topk=0)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=32,shuffle=False)
        return dataloader

    def _create_kwargs(self):
        kwargs = {
            "pretrained_model":torchvision.models.resnet18(pretrained=False, num_classes=100),
            "is_frozen": True,
            "use_saved_logits": True,
            "num_classes": 100
        }
        return kwargs

    def test_create(self):
        ''' test create a distiller

        :return:
        '''
        ################ frozen create ###################
        distiller = BasicDistiller(torchvision.models.resnet18(pretrained=True), is_frozen=True)
        for param in distiller.pretrained_model.parameters():
            assert param.requires_grad == False

        ################ unfrozen create ###################
        distiller = BasicDistiller(torchvision.models.resnet18(pretrained=True), is_frozen=False)
        for param in distiller.pretrained_model.parameters():
            assert param.requires_grad == True

    def prepare_logits(self):
        ''' test prepare logits

        :return:
        '''
        kwargs = self._create_kwargs()
        dataloader = self._create_dataloader(save_logits=True)
        distiller = BasicDistiller(**kwargs)
        distiller.prepare_logits(dataloader, 1, start_epoch=0)
        # assert os.path.exists("../datasets/logits_epoch0")

    def test_forward(self):
        ''' test forward

        :return:
        '''
        ########################## directly forward test ############################
        num_classes = 1000
        bath_size = 16
        distiller = BasicDistiller(torchvision.models.resnet18(pretrained=True), is_frozen=True)
        x = torch.zeros([bath_size,3,224,224])
        y = distiller(x)
        assert y.shape == torch.Size([bath_size,num_classes])
        ########################## load pretrain logits forward test ############################
        self.prepare_logits()
        kwargs = self._create_kwargs()
        dataloader = self._create_dataloader(save_logits=False)
        distiller = BasicDistiller(**kwargs)

        for (idx, (data, label)) in enumerate(dataloader):
            y = distiller(data)
            assert y.shape == torch.Size([data[0].shape[0],100])
            break

class TestKD:
    ''' Test KD
    
    '''
    def _get_kwargs(self):
        kwargs = {
            "pretrained_model": torchvision.models.resnet18(pretrained=True),
            "temperature": 4.0,
            "is_frozen": True,
        }
        return kwargs

    def test_loss(self):
        ''' test loss

        :return:
        '''
        kwargs = self._get_kwargs()
        distiller = KD(**kwargs)
        num_classes = 1000
        bath_size = 16
        logits1 = torch.ones([bath_size,num_classes])
        logits2 = torch.zeros([bath_size, num_classes])
        logits3 = torch.randn([bath_size,num_classes])
        logits4 = 1 - logits3
        assert torch.abs(distiller.loss(logits3, logits3)).item() <= 1e-4
        assert torch.abs(distiller.loss(logits1, logits2)).item() <= 1e-4
        assert torch.abs(distiller.loss(logits3, logits4)).item() >= 1


class TestDKD:
    ''' Test DKD
    
    '''
    def _get_kwargs(self):
        kwargs = {
            "pretrained_model": torchvision.models.resnet18(pretrained=True),
            "alpha": 1.0,
            "beta": 8.0,
            "temperature": 4.0,
            "warmup": 20,
            "is_frozen": True,
        }
        return kwargs

    def test_loss(self):
        ''' test loss

        :return:
        '''
        kwargs = self._get_kwargs()
        distiller = DKD(**kwargs)
        num_classes = 1000
        bath_size = 16
        logits1 = torch.ones([bath_size,num_classes])
        logits2 = torch.zeros([bath_size, num_classes])
        logits3 = torch.randn([bath_size,num_classes])
        target = torch.Tensor([random.randint(0,1000) for i in range(16)]).long()
        assert torch.abs(distiller.loss(logits3, logits3, epoch=20,target=target)).item() <= 1e-4
        assert torch.abs(distiller.loss(logits1, logits2, epoch=20,target=target)).item() <= 1e-4
        assert torch.abs(distiller.loss(logits1, logits3, epoch=20,target=target)).item() >= 1

# if __name__ == "__main__":
#     test = TestBasicDistiller()
#     test.test_forward()