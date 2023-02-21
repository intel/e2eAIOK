import os
import torch
from torchvision import datasets, transforms
from e2eAIOK.common.trainer.data.data_builder_cv import DataBuilderCV
from e2eAIOK.common.utils import check_mkdir
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

class DataBuilderImageNet(DataBuilderCV):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def prepare_dataset(self):
        """
            prepare ImageNet dataset
        """
        transform = self.build_transform(is_train = True)
        train_data_folder = check_mkdir(self.cfg.train_data_path)
        self.dataset_train = datasets.ImageFolder(train_data_folder, transform=transform)

        transform = self.build_transform(is_train = False)
        val_data_folder = check_mkdir(self.cfg.val_data_path)
        self.dataset_val = datasets.ImageFolder(val_data_folder, transform=transform)
        


    def build_transform(self, is_train):
        resize_im = "input_size" in self.cfg and self.cfg.input_size > 32
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=self.cfg.input_size,
                is_training=True,
                color_jitter=self.cfg.color_jitter,
                auto_augment=self.cfg.aa,
                interpolation=self.cfg.train_interpolation,
                re_prob=self.cfg.reprob,
                re_mode=self.cfg.remode,
                re_count=self.cfg.recount,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(
                    self.cfg.input_size, padding=4)
            return transform

        t = []
        if resize_im:
            size = int((256 / 224) * self.cfg.input_size)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(self.cfg.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)