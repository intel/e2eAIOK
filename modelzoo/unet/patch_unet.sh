#!/bin/bash

# generate patch
# git diff --cached > git.patch

get_original_model () {
    cp -r ../third_party/nnUNet .
}

apply_patch () {
    cd nnUNet
    git apply ../git.patch
    pip install -e .
    cd ..
}

get_original_model
apply_patch

