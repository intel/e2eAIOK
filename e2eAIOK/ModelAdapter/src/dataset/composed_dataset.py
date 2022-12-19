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

        ############ len list ############
        self._len_list = []
        for dataset in datasets:
            dataset_len = len(dataset)
            self._len_list.append(dataset_len)
            logging.info('dataset [%s] len = %s'%(str(dataset),dataset_len))
        ############ max len #############
        self._max_len = self._len_list[0]
        for _len in self._len_list:  # get the max len
            if _len > self._max_len:
                self._max_len = _len

    def __getitem__(self, index):
        if index >= self._max_len: # out range
            raise StopIteration

        data_list = []
        label_list = []
        for (dataset,_len) in zip(self._datasets,self._len_list):
            data,label = dataset[index % _len]
            data_list.append(data)
            label_list.append(label)

        return (data_list,label_list)

    def __len__(self):
        return self._max_len

    def __str__(self):
        output = 'ComposedDataset: length [%s]\n'% self.__len__()
        for (dataset, _len) in zip(self._datasets, self._len_list):
            output += '\tdataset[%s] length [%s]\n'%(str(dataset),_len)
        return output
