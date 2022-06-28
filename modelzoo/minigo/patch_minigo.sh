#!/bin/bash
get_original_model () {
    git clone --depth 1 --filter=blob:none --sparse https://github.com/mlcommons/training_results_v1.0 &&
    cd training_results_v1.0 &&
    git sparse-checkout set Intel/benchmarks/minigo/8-nodes-64s-8376H-tensorflow &&
    cd .. &&
    mv training_results_v1.0/Intel/benchmarks/minigo/8-nodes-64s-8376H-tensorflow/* ./ &&
    rm -rf training_results_v1.0
}

apply_patch () {
    patch -p5 < minigo.patch
}

get_original_model
apply_patch

