#!/bin/bash
port=${port:-12888}
token=${token:-""}

jupyter-lab --notebook-dir=./ --ip=0.0.0.0 --port=$port --no-browser --NotebookApp.token=$token --allow-root