#!/bin/bash
set -x

attempt_num=0
max_attempts=100

while [ $attempt_num -lt $max_attempts ]
do
  sh evaluate_mmlu.sh

  if [ $? -eq 0 ]; then
    # task success
    break
  else
    # task failed
    attempt_num=$[$attempt_num+1]
    echo "Task failed. Retrying..."
  fi
done

if [ $attempt_num -eq $max_attempts ]; then
  # all retries failed
  echo "Task failed after $max_attempts attempts."
  exit 1
fi