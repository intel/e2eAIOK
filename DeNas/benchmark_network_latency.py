'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os,sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv.train_image_classification as tic
import torch, time
import numpy as np


def __get_latency__(model, batch_size, resolution, channel, gpu, benchmark_repeat_times, fp16):
    if fp16:
        model = model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    the_image = torch.randn(batch_size, channel, resolution, resolution, dtype=dtype)
    model.eval()
    warmup_T = 3
    with torch.no_grad():
        for i in range(warmup_T):
            the_output = model(the_image)
        start_timer = time.time()
        for repeat_count in range(benchmark_repeat_times):
            the_output = model(the_image)

    end_timer = time.time()
    the_latency = (end_timer - start_timer) / float(benchmark_repeat_times) / batch_size
    return the_latency

def get_model_latency(model, batch_size, resolution, in_channels, gpu, repeat_times, fp16):
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    if fp16:
        model = model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    the_image = torch.randn(batch_size, in_channels, resolution, resolution, dtype=dtype,
                            device=device)

    model.eval()
    warmup_T = 3
    with torch.no_grad():
        for i in range(warmup_T):
            the_output = model(the_image)
        start_timer = time.time()
        for repeat_count in range(repeat_times):
            the_output = model(the_image)

    end_timer = time.time()
    the_latency = (end_timer - start_timer) / float(repeat_times) / batch_size
    return the_latency
