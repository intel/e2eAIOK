#!/usr/bin/env python
# coding: utf-8

# # Spark + Horovod for Burgerking

# In[1]:


import configargparse
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "4"

import pandas as pd
import numpy as np

import logging
import time
from datetime import datetime

import pyspark
import pyspark.sql.types as T
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import horovod.spark.torch as hvd
from horovod.spark.common.store import HDFSStore
from horovod.spark.common.backend import SparkBackend 


# ## Configuration

# In[2]:


p = configargparse.ArgParser(default_config_files=['../conf/burgerking.conf'])
p.add_argument("--spark-master-address", type=str)
p.add_argument("--spark-cores-max", type=int)
p.add_argument("--spark-executor-cores", type=int)
p.add_argument("--spark-executor-memory", type=str)
p.add_argument("--spark-default-parallelism", type=int)
p.add_argument("--spark-data-dir", type=str)
p.add_argument("--hdfs-store", type=str)
p.add_argument("--nics", type=str)
options, _ = p.parse_known_args()


# ## Spark Initialization

# In[3]:


spark = pyspark.sql.SparkSession.builder.master(options.spark_master_address)                                    .config("spark.cores.max", options.spark_cores_max)                                     .config("spark.executor.cores", options.spark_executor_cores)                                     .config("spark.executor.memory", options.spark_executor_memory)                                     .config("spark.default.parallelism", options.spark_default_parallelism)                                     .config("spark.sql.execution.arrow.enabled", "true")                                     .config("spark.executorEnv.PATH", os.environ['PATH'])                                     .getOrCreate()   
                                    


# ## Data preparation

# In[5]:


start = time.time()
df = spark.read.json(options.spark_data_dir)
train_df, test_df = df.randomSplit([0.999, 0.001], seed=100)
end = time.time()

prepare_time = end - start
print(f"# train data: {train_df.count()}")
print(f"# test data:  {test_df.count()}")
print(f"time:         {prepare_time:.2f}s")


# In[6]:


store = HDFSStore(options.hdfs_store)


# ## Model Definition

# In[7]:


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

        self.embeds_pluids = nn.Embedding(n_plus, 50, sparse=True)
        self.embeds_bkidx = nn.Embedding(n_bkids, 100, sparse=True)
        self.embeds_timeidx = nn.Embedding(n_time, 100, sparse=True)
        self.embeds_feelsBucket = nn.Embedding(n_feels, 100, sparse=True)
        self.embeds_weather = nn.Embedding(n_weather, 100, sparse=True)

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.hidden1 = nn.Linear(100, 100)
        self.hidden2 = nn.Linear(100, 1)
        self.flatten = nn.Flatten()

        self.drop_layer = nn.Dropout(p=0.3)
        self.fc = nn.Linear(fcn_input_size, fcn_output_size)


    def forward(self, pluids, timeidx, bkidx, weatheridx, feelsBucket):

        pluids = pluids.long()
        timeidx = timeidx.long()
        bkidx = bkidx.long()
        weatheridx = weatheridx.long()
        feelsBucket = feelsBucket.long()
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


# In[8]:


batch_size = 16000
num_epoch = 5
loss = nn.CrossEntropyLoss()
num_proc=options.spark_cores_max // options.spark_executor_cores
feature_cols = ["pluids", "timeidx", "bkidx", "weatheridx", "feelsBucket"]

model = BiRNN(50, 50, 1, 100, 522)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def f(data):
    #avoid omp resource competition
    os.environ["OMP_NUM_THREADS"] = "4"
    iter([os.environ["OMP_NUM_THREADS"]])


# ## Train

# In[9]:


backend = SparkBackend(num_proc, nics=[options.nics], use_mpi=True)
torch_estimator = hvd.TorchEstimator(backend=backend,
                                     store=store,
                                     model=model,
                                     optimizer=optimizer,
                                     loss=lambda input, target: loss(input, target.long()),
                                     feature_cols=feature_cols,
                                     input_shapes=[[-1, 5], [-1], [-1], [-1], [-1]],
                                     label_cols=['label'],
                                     batch_size=batch_size,
                                     epochs=num_epoch,
                                     verbose=2)
start = time.time()
torch_model = torch_estimator.fit(train_df).setOutputCols(['label_prob'])
end = time.time()
train_time = end - start
print(f"train time: {train_time}") 


# ## Eval

# In[12]:


# Evaluate the model on the held-out test DataFrame
pred_df = torch_model.transform(test_df)
argmax = udf(lambda v: float(np.argmax(v)), returnType=T.DoubleType())
pred_df = pred_df.withColumn('label_pred', argmax(pred_df.label_prob))
evaluator = MulticlassClassificationEvaluator(predictionCol='label_pred', labelCol='label', metricName='accuracy')
test_acc = evaluator.evaluate(pred_df)


# ## Statistics

# In[13]:


print(f"Data Preparation: {prepare_time:.2f}s\") 
print(f"Train Total: {train_time:.2f}s / epoch") 
print(f"Train Avg: {train_time/num_epoch:.2f}s / epoch") 
print(f'Test Acc: {test_acc:.2f}')

