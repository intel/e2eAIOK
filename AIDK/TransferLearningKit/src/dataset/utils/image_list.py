#!/usr/bin/python
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
from .loader import rgb_loader,l_loader
import logging

class ImageList(Dataset):
    ''' ImageList dataset

    '''
    def __init__(self, data_path,label_records,transform,img_mode='RGB'):
        ''' init method

        :param data_path: image dir
        :param label_records: label record list, each record format: "image_name image_label"
        :param transform: data transform
        :param img_mode: img mode, must one "RGB" or "L" (ignore letter case)
        '''
        imgs = [("%s/%s"%(data_path,val.split()[0]), int(val.split()[1])) for val in label_records]
        if len(imgs) == 0:
            logging.error("Found 0 images in subfolders of: %s" %data_path)
            raise (RuntimeError("Found 0 images in subfolders of: %s" %data_path))

        self.imgs = imgs
        self.transform = transform
        self.img_mode = img_mode

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
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def __str__(self):
        return 'ImageList: image num [%s], img_mode[%s], transform [%s]'%(
            len(self.imgs),self.img_mode,self.transform)