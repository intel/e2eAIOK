#!/bin/bash

input="/home/vmagent/app/hydro.ai/aidk_cicd.log"

strict_log_pattern=^.*Epoch" ".*," "Iteration" ".*to" ".*" "took" ".*" "s," ".*" "ms/iter," "loss" "is" ".*," "test" "took" ".*" "secs," "test_auc_is" ".*$
loose_log_pattern=^.*Epoch.*Iteration.*test_auc_is.*$

while IFS= read -r line
do
  if [[ "$line" =~ $strict_log_pattern ]]
  then
    echo "Log pattern passed!"
  fi
done <<< $(grep $loose_log_pattern $input)