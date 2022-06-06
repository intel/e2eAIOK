#!/bin/bash

get_original_model () {
    cp -r ../../third_party/mlperf_v1.0/NVIDIA/benchmarks/rnnt/implementations/pytorch/* ./
}

apply_patch () {
    git apply rnnt.patch
}

get_original_model
apply_patch