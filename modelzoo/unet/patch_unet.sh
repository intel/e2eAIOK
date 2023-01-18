#!/bin/bash

get_original_model () {
    cp -r ../third_party/nnUNet .
}

apply_patch () {
    patch -p0 < nnunet.patch
    cd nnUNet
    pip install -e .
}

get_original_model
apply_patch