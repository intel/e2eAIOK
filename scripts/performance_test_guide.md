AIOK Performance Test Guide

* How to perform test

prepare environment

``` bash
# run below script to all performance node
scripts/run_docker
```

Prepare performance kit

`git clone https://github.com/Intel-bigdata/Benchmarksuit/tree/master/pat-suite`

Put workload execution script in pat-suite/workload
``` bash
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --data_path "/home/vmagent/app/dataset/amazon_reviews" --model_name {name} 
```

Modify benchmark suit configuration, including node IP to collect system metric data, metric sample rate, which metric to collect etc.
``` bash
# Config nodes for system metrics data collection
ALL_NODES: host1 host2
# Replace with the path to the script that launches the job
CMD_PATH: workload/test.sh
# List of instruments to be used in the analysis
INSTRUMENTS: cpustat memstat netstat iostat vmstat jvms sysconfig emon
```
Launch workload execution and system metrics collection by simply execute `auto.sh $run_id`

Post data processing, generate profiling charts in excel and pdf
``` bash
cd PAT-post-processing
# Modify <source>$Benchmarksuit_path/results/result/instruments</source> in config.xml to your result dir
# Process system data
./pat-post-process.py
```

Expect result
``` bash
sar
emon
result
- score
- train time
- score by iterator curve
- best trained model path
- torch/tensorflow profiling
```

Performance Owner
tao1.he@intel.com
