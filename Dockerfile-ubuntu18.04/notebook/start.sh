#!/bin/bash
port=${port:-12888}
token=${token:-""}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

jupyter-lab --notebook-dir=/home/vmagent/app/e2eaiok --ip=0.0.0.0 --port=$port --no-browser --NotebookApp.token=$token --allow-root