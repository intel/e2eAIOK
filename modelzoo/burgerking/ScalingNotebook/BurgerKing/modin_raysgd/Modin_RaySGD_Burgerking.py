#!/usr/bin/env python
# coding: utf-8

# # Modin + RaySGD + Ray for BurgerKing 

# ## Configuration

# In[1]:


import configargparse
import os

p = configargparse.ArgParser(default_config_files=['../conf/burgerking.conf'])
p.add_argument("--ray-init-address", type=str)
p.add_argument("--data-prefix", type=lambda x: os.path.abspath(x))
options, _ = p.parse_known_args()


# ## Connect to Ray

# In[2]:


import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator

ray.shutdown()
ray.init(address=options.ray_init_address) 


# In[3]:


import logging
import time

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import modin.pandas as pd
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter


# In[4]:


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(filename='./drivethru_log',level=logging.DEBUG)


# ## Data Preparation 

# In[5]:


start = time.time()
batch_size = 16000
num_epoch = 1

df_list = []
num_files=10
for num in range(0,3):
    df = pd.read_json(os.path.join(options.data_prefix, f'{num}.json'), orient='columns', lines=True)
    df_list.append(df)
data = pd.concat(df_list)

end = time.time()
prepare_time =end - start
print(f"time: {prepare_time:.2f}s")


# In[6]:


n_plus=522
n_time=167
n_bkids=126
n_weather=35
n_feels=20

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


# ## Model Definition

# In[7]:


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
        
        x = plu_embed
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size) # 2 for bidirection 
        
        # Forward propagate gru
        gru_out, _ = self.gru(x, h0)
        ut = torch.tanh(self.hidden1(gru_out))
        et = self.hidden2(ut)
        att = F.softmax(torch.transpose(et, 2, 1), dim=-1)
        output= torch.matmul(att, gru_out)
        
        # flatten the output
        attention_output =self.flatten(output)
        context_features=torch.mul(attention_output,(1 + bkidx_embed + time_embed + weather_embed + feels_embed))
        ac1=F.relu(context_features)
        dropout1=self.drop_layer(ac1)
        output=self.fc(dropout1)
       
        return output


# ### Adding Custom Traing and Validation Operator for RNN

# In[8]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def model_creator(config):
    return BiRNN(50,50,5,100,1).to(device)

def optimizer_creator(model, config):
    """Returns optimizer."""
    return torch.optim.Adagrad(model.parameters(),lr=1e-2)

def data_creator(config):
    #Constructs Iterables for training and validation.
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
    return train_loader


# In[9]:


# Setup Customer Traing and Validation 


class RNNOperator(TrainingOperator):
    def setup(self, config):
        """Custom setup for this operator.

        Args:
            config (dict): Custom configuration value to be passed to
                all creator and operator constructors. Same as ``self.config``.
        """
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_idx = 0

    def train_batch(self, batch, batch_info):
        """Trains on one batch of data from the data creator.
        Args:
            batch: One item of the validation iterator.
            batch_info (dict): Information dict passed in from ``train_epoch``.

        Returns:
            A dict of metrics. Defaults to "loss" and "num_samples",
                corresponding to the total number of datapoints in the batch.
        """
        self.batch_idx += 1
        x, label = batch[:-1], batch[-1].squeeze()
        output = self.model(x).type(torch.FloatTensor).squeeze()
        self.optimizer.zero_grad()
        loss = self.criterion(output, label)
        loss = Variable(loss, requires_grad = True)
        
        # Backward and optimize
        if self.batch_idx % 1000 == 0:
            print(f'batch: {self.batch_idx}, loss: {loss.item()}')
        

        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item()
        }


# In[10]:


'''
Launches a set of actors which connect via distributed PyTorch and
coordinate gradient updates to train the provided model. If Ray is not
initialized, TorchTrainer will automatically initialize a local Ray
cluster for you. Be sure to run `ray.init(address="auto")` to leverage
multi-node training.
'''

def train_RNN(num_workers=1, use_gpu=False):
    trainer = TorchTrainer(
        model_creator=model_creator,
        data_creator=data_creator,
        optimizer_creator=optimizer_creator,
        loss_creator=torch.nn.MSELoss,
        #loss_creator=nn.BCELoss(),
        #loss_creator=torch.nn.CrossEntropyLoss(),
        training_operator_cls=RNNOperator,
        num_workers=num_workers,
        use_gpu=False,
        config={"batch_size": batch_size})
    
    for i in range(3):
        stats = trainer.train()
        print(f"Step: {i}, stats: {stats}")
        
    # print(trainer.validate())
    m = trainer.get_model()

    model = trainer.get_model()
    print(model.parameters())

    trainer.shutdown()
    print("success!")


# ## Train

# In[11]:


train_RNN(num_workers=2, use_gpu=False)

