#!/bin/bash

get_original_model () {
    cp -r ../../third_party/alibaba-ai-matrix/macro_benchmark/DIEN_TF2/ ai-matrix
}

apply_patch () {
    git apply dien.patch
    cat setvars.sh > tmp
    cat ai-matrix/script/train.py >> tmp
    mv tmp ai-matrix/script/train.py
}

get_original_model
apply_patch 
