## Download Data
1. Download data to criteo/raw_data
2. unzip with below cmdline to same folder
```
cd criteo/raw_data
for i in `seq 0 23`;do gzip -d day_$i.gz & done
ls
day_0  day_10  day_12  day_14  day_16  day_18  day_2   day_21  day_23  day_4  day_6  day_8
day_1  day_11  day_13  day_15  day_17  day_19  day_20  day_22  day_3   day_5  day_7  day_9
```

## Enter docker
```
docker run --shm-size=10g -it --privileged --network host --device=/dev/dri -v ${path_to_e2eaiok_dataset}:/home/vmagent/app/dataset -v `pwd`/:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ xuechendi/oneapi-aikit:hydro.ai /bin/bash
```

## Run DLRM data process
```
conda activate pytorch_mlperf
pip install pyrecdp

# spark service
cd /home/vmagent/app/e2eaiok/modelzoo/dlrm/data_processing/
cp spark-defaults.conf /home/spark-3.2.1-bin-hadoop3.2/conf/
mkdir -p /home/vmagent/app/e2eaiok/modelzoo/dlrm/data_processing/spark_local_dir
mkdir -p /home/mnt/applicationHistory
sh ./start_spark_service.sh

# trigger data processing
# dlrm data processing requires huge space, make sure there are at least 1.2T for /home/vmagent/app/e2eaiok/modelzoo/dlrm/data_processing/
rm -rf /home/vmagent/app/dataset/criteo/output/tmp/
python preprocessing.py

# convert
# to be added
```
