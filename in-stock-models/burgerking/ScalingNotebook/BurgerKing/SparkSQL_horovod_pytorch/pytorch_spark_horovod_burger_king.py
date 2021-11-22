import os
os.environ["OMP_NUM_THREADS"] = "16"

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

#spark://sr231:7077

        
if __name__ == "__main__":
    spark = pyspark.sql.SparkSession.builder.master("spark://sr231:7077")\
                                        .config("spark.cores.max", 96) \
                                        .config("spark.executor.cores", 16) \
                                        .config("spark.executor.memory", "120g") \
                                        .config("spark.default.parallelism", 96) \
                                        .config("spark.sql.execution.arrow.enabled", "true") \
                                        .config("spark.task.cpus", 1) \
                                        .getOrCreate()
    time1=time.time()
    df = spark.read.json("hdfs://sr231:9000/data")
  
    train_df, test_df = df.randomSplit([0.999, 0.001],seed=100)
    print("traindf count:",train_df.count())
    print("testdf count:",test_df.count())

    store = HDFSStore('hdfs://sr231:9000/tmp')
    time2=time.time()
    data_prepare_time=time2-time1
    print("data_prepare_time:",data_prepare_time)

    n_plus=522
    n_time=167
    n_bkids=126
    n_weather=35
    n_feels=20

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

    # train with SGD
    model = BiRNN(50, 50, 1, 100, 522)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()

    num_proc=6

    feature_cols = ["pluids", "timeidx", "bkidx", "weatheridx", "feelsBucket"]
    batch_size = 16000
    num_epoch = 5

    def f(data):
            # this is used to avoid omp resource competition
            os.environ["OMP_NUM_THREADS"] = "2"
            iter([os.environ["OMP_NUM_THREADS"]])
        
    #backend = SparkBackend(num_proc, nic="eth2")
    backend = SparkBackend(num_proc, nics=["eth2"], use_gloo=True)
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


    torch_model = torch_estimator.fit(train_df).setOutputCols(['label_prob'])
    time3 = time.time()
    print(f"Duration(train): {time3 - time2}") 

    
    #spark.range(0, 96, 96).rdd.mapPartitions(f).count()

    # Evaluate the model on the held-out test DataFrame
    start = time.time()
    pred_df = torch_model.transform(test_df)
    argmax = udf(lambda v: float(np.argmax(v)), returnType=T.DoubleType())
    pred_df = pred_df.withColumn('label_pred', argmax(pred_df.label_prob))
    evaluator = MulticlassClassificationEvaluator(predictionCol='label_pred', labelCol='label', metricName='accuracy')

    train_count=train_df.count()
    test_count=test_df.count()
    print("traindf count:",train_count)
    print("testdf count:",test_count)

    print(f"Duration(train): {time3 - time2}")
    print(f"Average epoch duration(train): {(time3 - time2)/5}") 
    print("data_prepare_time:",data_prepare_time)

    print('Test accuracy:', evaluator.evaluate(pred_df))
