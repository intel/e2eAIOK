#!/bin/bash
set -x

data=$1

value=`cat $data | grep \"acc_norm\" | cut -d: -f 2 | cut -d, -f 1 | awk '{sum += $1} END {printf "NR = %d,Average = %3.4f\n",NR,sum/NR}'`

echo $value