import pytest
import sys
import os
import sys
from torch.utils.data import Dataset
from e2eAIOK.ModelAdapter.src.dataset.composed_dataset import ComposedDataset

class NumberDataset(Dataset):
    ''' Toy Dataset to mimic real-world Dataset

    '''
    def __init__(self,feature_list,label_list):
        self._feature_list = feature_list
        self._label_list = label_list
        assert len(feature_list) == len(label_list)
    def __getitem__(self,index):
        return self._feature_list[index],self._label_list[index]
    def __len__(self):
        return len(self._feature_list)

class TestComposedDataset:
    ''' test ComposedDataset

    '''
    # range(feature_begin, feature_end) to generate feature
    feature_begin = 0
    feature_end = 10

    def test_length(self):
        ''' test ComposedDataset.__len__

        :return:
        '''
        for end1 in range(self.feature_begin,self.feature_end):
            length1 = end1 - self.feature_begin
            dataset1 = NumberDataset(range(self.feature_begin,end1),[0]*length1)

            for end2 in range(self.feature_begin,self.feature_end):
                length2 = end2 - self.feature_begin
                dataset2 = NumberDataset(range(self.feature_begin,end2), [1] * length2)

                composed_dataset = ComposedDataset(dataset1,dataset2)
                real_len = max(length1,length2) # max len
                assert len(composed_dataset) == real_len

    def test_iter(self):
        ''' test ComposedDataset.__getitem__

        :return:
        '''
        length1 = self.feature_end - self.feature_begin
        length2 = length1//4 # 1/4 length

        dataset1 = NumberDataset(range(self.feature_begin, self.feature_end), [0] * length1)
        dataset2 = NumberDataset(range(self.feature_begin, self.feature_begin + length2),[1] * length2)
        composed_dataset = ComposedDataset(dataset1, dataset2)
        for (step,(data,label)) in enumerate(composed_dataset):
            assert data == [step + self.feature_begin, self.feature_begin + step % length2]
            assert label == [0, 1]