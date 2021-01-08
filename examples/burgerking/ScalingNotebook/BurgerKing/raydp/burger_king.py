#!/usr/bin/env python
# coding: utf-8

# # RayDP for Burgerking

# In[ ]:


import configargparse
import os

import databricks.koalas as ks
import ray
from ray.util.sgd.torch.torch_trainer import TorchTrainer

import raydp.spark.context as context
from raydp.spark.torch_sgd import TorchEstimator
from raydp.spark.utils import random_split

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict


# # Configuration

# In[ ]:


p = configargparse.ArgParser(default_config_files=['../conf/burgerking.conf'])
p.add_argument("--spark-home", type=str)
p.add_argument("--ray-address", type=str)
p.add_argument("--ray-node-ip", type=str)
p.add_argument("--ray-passwd", type=str)
p.add_argument("--ray-data-path", type=str)

options, _ = p.parse_known_args()


# # Ray Init

# In[ ]:


# add spark home into the env
os.environ["SPARK_HOME"] = options.spark_home
GB = 1024 * 1024 * 1024

# connect to ray cluster
ray.init(address=options.ray_address, node_ip_address=options.ray_node_ip, redis_password=options.ray_passwd)

# init spark context
context.init_spark(app_name="Burger King",
                   num_executors=2,
                   executor_cores=10,
                   executor_memory=int(40 * GB))


# # Load Data

# In[ ]:


# data processing with koalas
df: ks.DataFrame = ks.read_json(options.ray_data_path)
train_df, test_df = random_split(df, [0.7, 0.3])


# # Model

# In[ ]:


n_plus = 522
n_time = 167
n_bkids = 126
n_weather = 35
n_feels = 20

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


# # Train

# In[ ]:


# train with SGD
model = BiRNN(50, 50, 5, 100, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()

estimator = TorchEstimator(num_workers=2,
                           model=model,
                           optimizer=optimizer,
                           loss=loss,
                           feature_columns=["pluids", "timeidx", "bkidx", "weatheridx", "feelsBucket"],
                           feature_shapes=[5, 0, 0, 0, 0],
                           feature_types=[torch.long, torch.long, torch.long, torch.long, torch.long],
                           label_column="label",
                           label_type=torch.long,
                           batch_size=100,
                           num_epochs=10)

estimator.fit(train_df)


# # Evaluation

# In[ ]:


estimator.evaluate(test_df)


# In[ ]:


print(estimator.get_model())


# In[ ]:


estimator.shutdown()
context.stop_spark()
ray.shutdown()

