# How to run

```
# cp stack-overflow folder to ${current_path}/recdp/examples/python_tests/haystack_sod/

# run below codes to all physical nodes
# Fix the recdp path to align with your env

docker run -it --net=host -v /mnt/nvme2/chendi/BlueWhale/recdp:/home/vmagent/app/recdp -v /mnt/nvme2/chendi/BlueWhale/applications.ai.appliedml.haystack/:/home/haystack -v /mnt/nvme2/chendi/BlueWhale/applications.ai.appliedml.haystack/haystack/:/home/user/haystack xuechendi/haystack-cpu-recdp /bin/bash

# after enter docker, start spark master and workers

vim ${SPARK_HOME}/conf/workers
# add all workers hostname
# Example
sr602
sr603
sr604
...

# start spark master on one of your node
${SPARK_HOME}/sbin/start-master.sh

# start spark workers on all nodes
${SPARK_HOME}/sbin/start-worker.sh spark://${master_hostname}:7077

# run gen sod script
cd /home/vmagent/app/recdp/examples/python_tests/haystack_sod/

# doing spark configuration in gen_sod_to_documentstore.py
EMBEDDING_PARALLELISM = 2  # num_nodes * 2
NUMCORES_PER_SOCKET = 52
EMBEDDING_BATCH_SIZE = 32768

python gen_sod_to_documentstore.py
```
