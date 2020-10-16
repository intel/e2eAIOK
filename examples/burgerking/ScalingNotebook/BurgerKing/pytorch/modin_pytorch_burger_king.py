import os
import ray
ray.shutdown()
ray.init(address="10.1.0.131:6379", node_ip_address="10.1.0.131", redis_password="123") 

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import boto3
import modin.pandas as pd

import numpy as np
import s3fs
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import logging
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime

import horovod.torch as hvd


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(filename='./drivethru_log',level=logging.DEBUG)

bucket = ...
prefix = '/home/sparkuser/jupyter/Bin/BlueWhale-POC/data/'
model_prefix = ...
batch_size = 20
num_epoch = 1
#num_gpus = range(8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_checkpoint_interval = 5


time1=datetime.now()
df_list = []
num_files=10
for num in range(0,1):
    df = pd.read_json(prefix+str(num)+".json", orient='columns', lines=True)
    df_list.append(df)

data = pd.concat(df_list)


n_plus=522
n_time=167
n_bkids=126
n_weather=35
n_feels=20

data['pluids1']=data['pluids'].str.get(0)
data['pluids2']=data['pluids'].str.get(1)
data['pluids3']=data['pluids'].str.get(2)
data['pluids4']=data['pluids'].str.get(3)
data['pluids5']=data['pluids'].str.get(4)


#convert pluid lists to pytorch long tensor 

pluids=data[['pluids1','pluids2','pluids3','pluids4','pluids5']]
timeidx=data[['timeidx']]
bkidx=data[['bkidx']]
weatheridx=data[['weatheridx']]
feelsBucket=data[['feelsBucket']]
target=data[['label']]

pluids_tensor=torch.LongTensor(pluids.values)
bkidx_tensor=torch.LongTensor(bkidx.values)
timeidx_tensor=torch.LongTensor(timeidx.values)
weatheridx_tensor=torch.LongTensor(weatheridx.values)
feelsBucket_tensor=torch.LongTensor(feelsBucket.values)
target_tensor=torch.LongTensor(target.values)

Train_Dataset=TensorDataset(pluids_tensor,bkidx_tensor,timeidx_tensor,weatheridx_tensor,feelsBucket_tensor,target_tensor)

train_loader = torch.utils.data.DataLoader(dataset=Train_Dataset,
                                           batch_size=batch_size,shuffle=True)
time2=datetime.now()
data_prepare_time=time2-time1
print("data load time:")
print(data_prepare_time)

def plu_embedding(pluids):
    embeds_pluids = nn.Embedding(n_plus, 50)

    pluids = Variable(pluids)
    plu_embed = embeds_pluids(pluids)
    return plu_embed
def bkidx_embedding(bkidx):
    embeds_bkidx = nn.Embedding(n_bkids, 100)

    bkidx = Variable(bkidx)
    bkidx_embed = embeds_bkidx(bkidx)
    return bkidx_embed
def timeidx_embedding(timeidx):
    embeds_timeidx = nn.Embedding(n_time, 100)

    timeidx = Variable(timeidx)
    time_embed = embeds_timeidx(timeidx)
    return time_embed
def feels_embedding(feelsBucket):
    embeds_feelsBucket = nn.Embedding(n_feels,100)

    feelsBucket = Variable(feelsBucket)
    feels_embed = embeds_feelsBucket(feelsBucket)
    return feels_embed
def weather_embedding(weather):
    embeds_weather = nn.Embedding(n_weather, 100)

    weather = Variable(weather)
    weather_embed = embeds_weather(weather)
    return weather_embed


# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,fcn_input_size,fcn_output_size):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        
        self.hidden1 = nn.Linear(100,100)
        self.hidden2 = nn.Linear(100,1)
        self.flatten=nn.Flatten()
        
        self.fcn_input_size=fcn_input_size
        self.fcn_output_size=fcn_output_size
        
        self.drop_layer=nn.Dropout(p=0.3)
        self.fc=nn.Linear(fcn_input_size,fcn_output_size)
        
    def forward(self,pluids,bkidx,timeidx,weatheridx,feelsBucket):
        # Set initial states
        
        plu_embed=plu_embedding(pluids)
        bkidx_embed=bkidx_embedding(bkidx)
        time_embed=timeidx_embedding(timeidx)
        weather_embed=weather_embedding(weatheridx)
        feels_embed=feels_embedding(feelsBucket)

        x=plu_embed
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size) # 2 for bidirection 
        #c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        # Forward propagate gru
        #gru_out, _ = self.gru(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        gru_out, _ = self.gru(x, h0)
        ut = F.tanh(self.hidden1(gru_out))
        # et shape: [batch_size, seq_len, att_hops]
        et = self.hidden2(ut)

        # att shape: [batch_size,  att_hops, seq_len]
        att = F.softmax(torch.transpose(et, 2, 1))
        
        # output shape [batch_size, att_hops, embedding_width]
        output= torch.matmul(att, gru_out)
        
        #flatten the output
        attention_output =self.flatten(output)
        
        context_features=torch.mul(attention_output,(1 + bkidx_embed + time_embed + weather_embed + feels_embed))
        

        ac1=F.relu(context_features)
        dropout1=self.drop_layer(ac1)
        output=self.fc(dropout1)
        return output

optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

optimizer= hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression)
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)


model=BiRNN(50, 50, 1,100,522).to(device)
learning_rate=0.01

criterion = nn.CrossEntropyLoss()


time3=datetime.now()
total_step = len(train_loader)
print("total step")
for epoch in range(num_epoch):
    for batchidx,(pluids,bkidx,timeidx,weatheridx,feelsBucket,target) in enumerate(train_loader):    
           
       
        output=model(pluids,bkidx,timeidx,weatheridx,feelsBucket)
        label=Variable(target)
        
        loss = criterion(output, label)
        
        # Backward and optimize
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        print('Epoch: ', epoch,'Step [{}/{}]'.format(batchidx,total_step),' loss: ', loss.item())

time4=datetime.now()
model_train_time=time4-time3
print("model train time")
print(model_train_time)

## Evaluation process

test_df_list = []
num_files=10
for num in range(7,10):
    df = pd.read_json(prefix+str(num)+".json", orient='columns', lines=True)
    test_df_list.append(df)

test_data = pd.concat(test_df_list)

test_data['pluids1']=test_data['pluids'].str.get(0)
test_data['pluids2']=test_data['pluids'].str.get(1)
test_data['pluids3']=test_data['pluids'].str.get(2)
test_data['pluids4']=test_data['pluids'].str.get(3)
test_data['pluids5']=test_data['pluids'].str.get(4)

test_pluids=test_data[['pluids1','pluids2','pluids3','pluids4','pluids5']]
test_timeidx=test_data[['timeidx']]
test_bkidx=test_data[['bkidx']]
test_weatheridx=test_data[['weatheridx']]
test_feelsBucket=test_data[['feelsBucket']]
test_target=test_data[['label']]

test_pluids_tensor=torch.LongTensor(test_pluids.values)
test_bkidx_tensor=torch.LongTensor(test_bkidx.values)
test_timeidx_tensor=torch.LongTensor(test_timeidx.values)
test_weatheridx_tensor=torch.LongTensor(test_weatheridx.values)
test_feelsBucket_tensor=torch.LongTensor(test_feelsBucket.values)
test_target_tensor=torch.LongTensor(test_target.values)

test_plu_embed=plu_embedding(test_pluids_tensor)
test_bkidx_embed=bkidx_embedding(test_bkidx_tensor)
test_time_embed=timeidx_embedding(test_timeidx_tensor)
test_weather_embed=weather_embedding(test_weatheridx_tensor)
test_feels_embed=feels_embedding(test_feelsBucket_tensor)


test_Dataset=TensorDataset(test_pluids_tensor,test_bkidx_tensor,test_timeidx_tensor,test_weatheridx_tensor,test_feelsBucket_tensor,test_target_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_Dataset,
                                           batch_size=batch_size,shuffle=True)
                                           
correct=0
for batchidx,(pluids,bkidx,timeidx,weatheridx,feelsBucket,target) in enumerate(test_loader):    
           
        label = Variable(target)

        prediction=model(pluids,bkidx,timeidx,weatheridx,feelsBucket)
        correct+=(prediction==label).sum().float()

total=len(test_target)

print("Accuracy:%f"%(correct/total).cpu().detach().data.numpy())
print(model_train_time)