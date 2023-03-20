#!/bin/bash

get_original_model () {
    cp -r ../../third_party/alibaba-ai-matrix/macro_benchmark/DIEN_TF2/ ai-matrix
}

apply_patch () {
    cp dien.patch ai-matrix
    cd ai-matrix
    git init --initial-branch=main
    git add *
    git apply dien.patch
    cd ..
}

get_original_model
apply_patch 
