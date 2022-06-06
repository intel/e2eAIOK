#!/bin/bash
get_original_model () {
    git init &&
    git remote add origin_bert https://github.com/IntelAI/models.git &&
    git pull origin_bert master
}

apply_patch () {
    git apply --stat bert.patch &&
    git apply --check bert.patch &&
    git apply bert.patch
}

get_original_model
apply_patch