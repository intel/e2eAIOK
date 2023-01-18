from torch.utils.data import Dataset
from e2eAIOK.common.trainer.data.data_utils.loader import rgb_loader,l_loader
import logging
import os

class Office31(Dataset):
    ''' Office31 Dataset

    '''
    def __init__(self, data_path, label_map, transform,img_mode='RGB'):
        ''' Init method

        :param data_path: img data location
        :param label_map: map from label name to label id
        :param transform: data transform
        :param img_mode: img mode
        '''
        self.data_path = data_path
        self._label_map = label_map
        self.transform = transform
        self.img_mode = img_mode
        self._getFileAndImgPath(data_path)

        if img_mode.upper() == 'RGB':
            self.loader = rgb_loader
        elif img_mode.upper()  == 'L':
            self.loader = l_loader
        else:
            logging.error("mode muse be one of: 'RGB','L' (ignore letter case), but found :%s"%img_mode)
            raise ValueError("mode muse be one of: 'RGB','L' (ignore letter case), but found :%s"%img_mode)

    def _getFileAndImgPath(self,root_path):
        self.imgs = []

        for label_name in sorted(os.listdir(root_path)):
            label_path = "%s/%s"%(root_path,label_name)
            label_id = self._label_map[label_name]
            for img_name in os.listdir(label_path):
                img_path = "%s/%s"%(label_path,img_name)
                self.imgs.append((img_path,label_id))

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

    def __str__(self):
        return 'Office31: image num [%s], img_mode[%s], transform [%s]'%(
            len(self.imgs),self.img_mode,self.transform)
