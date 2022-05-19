#!/bin/bash

strict_log_pattern=^.*Epoch" ".*," "Iteration" ".*to" ".*" "took" ".*" "s," ".*" "ms/iter," "loss" "is" ".*," "test" "took" ".*" "secs," "test_auc" ".*$
loose_log_pattern=^.*Epoch.*Iteration.*test_auc.*$

log_flag=0
while IFS= read -r line
do
  if [[ "$line" =~ $strict_log_pattern ]]
  then
    log_flag=1
  fi
done <<< $(grep $loose_log_pattern $LOG_FILE)

profile_pattern=^.*Self.*CPU.*time.*total.*$
torch_profile_flag=0
while IFS= read -r line
do
  if [[ "$line" =~ $profile_pattern ]]
  then
    torch_profile_flag=1
  fi
done <<< $(grep $profile_pattern $LOG_FILE)

if [[ $log_flag == 1 && $torch_profile_flag == 1 ]]
then
  echo "Log pattern passed!"
fi