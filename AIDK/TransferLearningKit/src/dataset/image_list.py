#!/usr/bin/python
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
from .utils import rgb_loader,l_loader
import logging

class ImageList(Dataset):
    ''' ImageList dataset

    '''
    def __init__(self, data_path,label_records, data_transform=None, img_mode='RGB'):
        ''' init method

        :param data_path: image dir
        :param label_records: label record list, each record format: "image_name image_label"
        :param data_transform: transform on image
        :param img_mode: img mode, must one "RGB" or "L" (ignore letter case)
        '''
        imgs = [("%s/%s"%(data_path,val.split()[0]), int(val.split()[1])) for val in label_records]
        if len(imgs) == 0:
            logging.error("Found 0 images in subfolders of: %s" %data_path)
            raise (RuntimeError("Found 0 images in subfolders of: %s" %data_path))

        self.imgs = imgs
        self.img_mode = img_mode
        self.data_transform = data_transform
        if data_transform is not None:
            logging.debug("Applying data_transform on %s" % data_path)

        if img_mode.upper() == 'RGB':
            self.loader = rgb_loader
        elif img_mode.upper()  == 'L':
            self.loader = l_loader
        else:
            logging.error("mode muse be one of: 'RGB','L' (ignore letter case), but found :%s"%img_mode)
            raise ValueError("mode muse be one of: 'RGB','L' (ignore letter case), but found :%s"%img_mode)

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)
        if self.data_transform is not None:
            img = self.data_transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

    def __str__(self):
        return 'ImageList: image num [%s], data_transform [%s], img_mode [%s]'%(
            len(self.imgs),self.data_transform,self.img_mode
        )