#!/bin/bash

get_original_model () {
    git init &&
    git config remote.origin_bert.url >&- || git remote add origin_bert https://github.com/IntelAI/models.git &&
    rm README.md && git pull origin_bert r2.5
}

apply_patch () {
    git apply --stat bert.patch &&
    git apply --check bert.patch &&
    git apply --whitespace=nowarn bert.patch
}

get_original_model
apply_patch
