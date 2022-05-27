#!/bin/bash

get_original_model () {
    cp -r ../../third_party/DeepLearningExamples/TensorFlow2/Recommendation/WideAndDeep/* ./
}

apply_patch () {
    git apply wnd.patch
}

get_original_model
apply_patch