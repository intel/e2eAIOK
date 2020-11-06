import ray

ray.init(address="auto", node_ip_address=None, redis_password="123")


import numpy as np

import ray
from ray.util.sgd.torch.torch_trainer import TorchTrainer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from typing import Dict

import os

import modin.pandas as pd
from modin_dataset import ModinDataset, ModinDatasetSampler
import time


os.environ["OMP_NUM_THREADS"] = "16"
os.environ["NCCL_TIMEOUT_S"] = "30"
os.environ["GLOO_SOCKET_IFNAME"] = "eth2" 

time1=time.time()


batch_size = 16000
num_epoch = 5

df_list = []
prefix='/mnt/DP_disk1/buger_king_data/sample_data/'
#prefix='/mnt/DP_disk1/buger_king_data/sample_6_file/'
#filelist = '/mnt/DP_disk1/buger_king_data/filelist.txt'
#filelist = '/mnt/DP_disk1/buger_king_data/filelist_198.txt'
#filelist = '/mnt/DP_disk1/buger_king_data/tmp/file_36.txt'

print('loading files')

with open(filelist) as f:
    content = f.readlines()
    
for filename in content:
    df = pd.read_json(prefix+str(filename.strip()), dtype=False, orient='columns', lines=True)
    df_list.append(df)
print(filelist)
df = pd.concat(df_list)
'''
for num in range(0,6):
    print(prefix+str(num)+".json")
    df = pd.read_json(prefix+str(num)+".json", dtype=False, orient='columns', lines=True)
    df_list.append(df)

df = pd.concat(df_list)
'''
print('train_test_split')
split_begin_time=time.time()
train_df, test_df = train_test_split(df, test_size=0.001, random_state=100, shuffle=False)
time2=time.time()

print("data load time:",split_begin_time-time1)
print("split data time:",time2-split_begin_time)

train_dataset = ModinDataset(df=df,
                       feature_columns=["pluids", "timeidx", "bkidx", "weatheridx", "feelsBucket"],
                       feature_shapes=[5, 0, 0, 0, 0],
                       feature_types=[torch.long, torch.long, torch.long, torch.long, torch.long],
                       label_column="label",
                       label_type=torch.long)

test_dataset = ModinDataset(df=test_df,
                       feature_columns=["pluids", "timeidx", "bkidx", "weatheridx", "feelsBucket"],
                       feature_shapes=[5, 0, 0, 0, 0],
                       feature_types=[torch.long, torch.long, torch.long, torch.long, torch.long],
                       label_column="label",
                       label_type=torch.long)

time3=time.time()
print("create modin dataset time:",time3-time2)
print("data prepare time:",time3-time1)

n_plus=544
n_time=168
n_bkids=129
n_weather=42
n_feels=21

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, fcn_input_size, fcn_output_size):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embeds_pluids = nn.Embedding(n_plus, 50)
        self.embeds_bkidx = nn.Embedding(n_bkids, 100)
        self.embeds_timeidx = nn.Embedding(n_time, 100)
        self.embeds_feelsBucket = nn.Embedding(n_feels, 100)
        self.embeds_weather = nn.Embedding(n_weather, 100)
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        self.hidden1 = nn.Linear(100, 100)
        self.hidden2 = nn.Linear(100, 1)
        self.flatten = nn.Flatten()
        
        self.drop_layer = nn.Dropout(p=0.3)
        self.fc = nn.Linear(fcn_input_size, fcn_output_size)
        
    def forward(self, pluids, timeidx, bkidx, weatheridx, feelsBucket):
        plu_embed = self.embeds_pluids(pluids)
        bkidx_embed = self.embeds_bkidx(bkidx)
        time_embed = self.embeds_timeidx(timeidx)
        weather_embed = self.embeds_weather(weatheridx)
        feels_embed = self.embeds_feelsBucket(feelsBucket)

        x = plu_embed

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size) # 2 for bidirection 
        # Forward propagate gru
        gru_out, _ = self.gru(x, h0)
        ut = torch.tanh(self.hidden1(gru_out))
        # et shape: [batch_size, seq_len, att_hops]
        et = self.hidden2(ut)

        # att shape: [batch_size,  att_hops, seq_len]
        att = F.softmax(torch.transpose(et, 2, 1))
        
        # output shape [batch_size, att_hops, embedding_width]
        output = torch.matmul(att, gru_out)
        
        # flatten the output
        attention_output = self.flatten(output)
        context_features = torch.mul(attention_output,(1 + bkidx_embed + time_embed + weather_embed + feels_embed))
        ac1 = F.relu(context_features)
        
        dropout = self.drop_layer(ac1)
        output = self.fc(dropout)
        return output
    



def create_mode(config: Dict):
    model = BiRNN(50, 50, 1, 100, 546)
    return model

def data_creator(config: Dict):
    train_sampler = ModinDatasetSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_sampler = ModinDatasetSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, sampler=test_sampler)
    
    return train_loader,test_loader

def optimizer_creator(model, config: Dict):
    return torch.optim.Adam(model.parameters(), lr=0.01)
print("********************START TRAIN***********************")
trainer = TorchTrainer(model_creator=create_mode,
                       data_creator=data_creator,
                       optimizer_creator=optimizer_creator,
                       loss_creator=nn.CrossEntropyLoss,
                       num_workers=6,
                       add_dist_sampler=False)
for i in range(5):
    stats = trainer.train()
    print(f"Step: {i}, stats: {stats}")
time4=time.time()
print("model train time:",time4-time3)
print("*******************END TRAIN, START VALIDATION***********")
print(trainer.validate())
print("data load time:",split_begin_time-time1)
print("split data time:",time2-split_begin_time)
print("create modin dataset time:",time3-time2)
print("data prepare time:",time3-time1)
print("model train time:",time4-time3)
