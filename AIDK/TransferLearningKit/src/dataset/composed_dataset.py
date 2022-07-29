#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 7/28/2022 9:43 AM

from torch.utils.data import Dataset
import logging

class ComposedDataset(Dataset):
    ''' Composed Dataset: compose a series of datasets. For example, target dataset and source dataset

    '''
    def __init__(self,*datasets):
        ''' Init method

        :param datasets: a series of datasets
        '''
        self._datasets = datasets
        self._len_list = []
        for dataset in datasets:
            dataset_len = len(dataset)
            self._len_list.append(dataset_len)
            logging.info('dataset [%s] len = %s'%(str(dataset),dataset_len))

    def __getitem__(self, index):
        data_list = []
        label_list = []
        for (dataset,_len) in zip(self._datasets,self._len_list):
            data,label = dataset[index % _len]
            data_list.append(data)
            label_list.append(label)

        return (data_list,label_list)

    def __len__(self):
        max_len = self._len_list[0]
        for _len in self._len_list: # get the max len
            if _len > max_len:
                max_len = _len
        return max_len

    def __str__(self):
        output = 'ComposedDataset: length [%s]\n'% self.__len__()
        for (dataset, _len) in zip(self._datasets, self._len_list):
            output += '\tdataset[%s] length [%s]\n'%(str(dataset),_len)
        return output
