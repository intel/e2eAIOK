#!/usr/bin/env python
# coding: utf-8

# # Pytorch Single Node BurgerKing

# In[1]:


import configargparse
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
from sklearn.model_selection import train_test_split
import time


# ## Configuration

# In[2]:


p = configargparse.ArgParser(default_config_files=['../conf/burgerking.conf'])
p.add_argument("--data-prefix", type=lambda x: os.path.abspath(x))
options, _ = p.parse_known_args()
batch_size = 16000
num_epoch = 5


# ## Data Preparation

# In[3]:


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(filename='./drivethru_log',level=logging.DEBUG)
prefix=options.data_prefix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start = time.time()
df_list = []
for num in range(0,6):
    df = pd.read_json(os.path.join(prefix, f"{num}.json"), orient='columns', lines=True)
    df_list.append(df)

raw_data = pd.concat(df_list)

end = time.time()
load_data_time = end - start
print(f"load data time: {load_data_time:.2f}s")


# In[4]:


start = end
data, test = train_test_split(raw_data, test_size=0.001, random_state=100)
end = time.time()
split_data_time = end - start
print(f"split data time: {split_data_time:.2f}s")


# In[5]:


n_plus = 522
n_time = 167
n_bkids = 126
n_weather = 35
n_feels = 20

data = data[["pluids", "timeidx", "bkidx", "weatheridx", "feelsBucket", "label"]]
train_tensors = [
    torch.LongTensor(data['pluids'].tolist()),
    torch.tensor(data[["timeidx"]].values),
    torch.tensor(data[["bkidx"]].values),
    torch.tensor(data[["weatheridx"]].values),
    torch.tensor(data[["feelsBucket"]].values),
    torch.tensor(data[["label"]].values),
]
train_dataset = TensorDataset(*train_tensors)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)


# ## Model Definition

# In[6]:


# Bidirectional recurrent neural network (many-to-one)

# below is model is built following MXNet's example 
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,fcn_input_size,fcn_output_size):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embeds_pluids = nn.Embedding(n_plus, 50)
        self.embeds_bkidx = nn.Embedding(n_bkids, 100)
        self.embeds_timeidx = nn.Embedding(n_time, 100)
        self.embeds_feelsBucket = nn.Embedding(n_feels, 100)
        self.embeds_weather = nn.Embedding(n_weather, 100)
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        self.hidden1 = nn.Linear(100,100)
        self.hidden2 = nn.Linear(100,1)
        
        self.flatten=nn.Flatten()
        
        self.fcn_input_size=fcn_input_size
        self.fcn_output_size=fcn_output_size
        
        self.drop_layer=nn.Dropout(p=0.3)
        self.fc=nn.Linear(fcn_input_size,fcn_output_size)
        
    def forward(self, x):

        # Set initial states
        pluids, timeidx, bkidx, weatheridx, feelsBucket = x
        plu_embed = self.embeds_pluids(pluids.type(torch.LongTensor)).squeeze()
        bkidx_embed = self.embeds_bkidx(bkidx.type(torch.LongTensor)).squeeze()
        time_embed = self.embeds_timeidx(timeidx.type(torch.LongTensor)).squeeze()
        weather_embed = self.embeds_weather(weatheridx.type(torch.LongTensor)).squeeze()
        feels_embed = self.embeds_feelsBucket(feelsBucket.type(torch.LongTensor)).squeeze()
        
        x=plu_embed
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size) # 2 for bidirection 

        # Forward propagate gru
        gru_out, _ = self.gru(x, h0)
        ut = torch.tanh(self.hidden1(gru_out))
        et = self.hidden2(ut)
        att = F.softmax(torch.transpose(et, 2, 1), dim=-1)
        output= torch.matmul(att, gru_out)
        
        #flatten the output
        attention_output =self.flatten(output)
        context_features=torch.mul(attention_output,(1 + bkidx_embed + time_embed + weather_embed + feels_embed))
        ac1=F.relu(context_features)
        dropout1=self.drop_layer(ac1)
        output=self.fc(dropout1)
       
        return output


# ## Train

# In[7]:


model=BiRNN(50, 50, 1,100,522).to(device)
learning_rate=0.01

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)


start = time.time() 
print(f"total steps: {len(train_loader)}")

for epoch in range(num_epoch):
    epoch_start = time.time()
    for batchidx, batch in enumerate(train_loader):    
        
        x, label = batch[:-1], batch[-1].squeeze()
        output = model(x)        
        loss = criterion(output, label)
        
        # Backward and optimize
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}\t Step [{batchidx}/{len(train_loader)}]\t loss: {loss.item()}')
        
    epoch_end = time.time()
    
    print(f"Epoch time: {epoch_end - epoch_start:.2f}s")
    
end = time.time()
train_time = end - start

print(f"model train time: {train_time:.2f}")


# ## Eval

# In[8]:


test_df_list = []
for num in range(1,3):
    df = pd.read_json(os.path.join(prefix, f"{num}.json"), orient='columns', lines=True)
    test_df_list.append(df)

test_data = pd.concat(test_df_list)
test_data = test_data[["pluids", "timeidx", "bkidx", "weatheridx", "feelsBucket", "label"]]


# In[9]:


test_tensors = [
    torch.LongTensor(test_data['pluids'].tolist()),
    torch.tensor(test_data[["timeidx"]].values),
    torch.tensor(test_data[["bkidx"]].values),
    torch.tensor(test_data[["weatheridx"]].values),
    torch.tensor(test_data[["feelsBucket"]].values),
    torch.tensor(test_data[["label"]].values),
]

test_dataset=TensorDataset(*test_tensors)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
                               
        
correct=0
total = 0
for batchidx, batch in enumerate(test_loader):    
           
        x, label = batch[:-1], batch[-1]
        prediction=model(x)
        
        prediction = prediction.argmax(dim=1)
        correct+=(prediction==label.flatten()).sum().float()
        total += len(label)

print(f"Accuracy:{(correct/total).cpu().detach().data * 100:.2f}%")

