#!/bin/bash

# generate patch
# cp -r ../third_party/nnUNet/ .
# diff -urN nnUNet/ nnUNetNew > nnunet.patch

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

