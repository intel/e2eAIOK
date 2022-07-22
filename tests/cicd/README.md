# AIDK CICD Integration Tests and Unit Tests

AIDK Integration Tests(Via BATS framework) provide end-to-end testing of AIDK and built-in workflows (pipeline_test/DIEN/WnD/DLRM/RNNT/MiniGo/BERT), which simulate the real world usage scenarios of AIDK.\
AIDK Unit Tests(Via Pytest framework) verify the code is working as expected by artificially created data of (input,expectation) pair.


## How to create a test script for your workload

Create bash script named `jenkins_${model}_test.sh` under `tests/cicd`, then define your model specific variables `MODEL_NAME`, `DATA_PATH`, `CONF_FILE`.

## Prepare dataset

copy model specific dataset to `/mnt/DP_disk1/dataset/{model_specific_dataset}`

## Test scripts that can be run manually by developer

The easiest way to run test scripts is with Docker.\
Firstly, build AIDK docker image.
```
$ cd Dockerfile-ubuntu18.04
$ docker build -t aidk-tensorflow . -f DockerfileTensorflow
$ docker build -t aidk-pytorch . -f DockerfilePytorch
$ docker build -t aidk-pytorch110 . -f DockerfilePytorch110
```

Then, run test script for specific workflow.\
For pipeline_test:
```
$ docker run --rm --privileged --network host --device=/dev/dri -v /root/cicd_logs:/home/vmagent/app/cicd_logs -v /mnt/DP_disk1/dataset:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ aidk-tensorflow /bin/bash -c "SIGOPT_API_TOKEN=${SIGOPT_API_TOKEN} USE_SIGOPT=${USE_SIGOPT} . /home/vmagent/app/hydro.ai/tests/cicd/jenkins_pipeline_test.sh"
```
For DIEN:
```
$ docker run --rm --privileged --network host --device=/dev/dri -v /root/cicd_logs:/home/vmagent/app/cicd_logs -v /mnt/DP_disk1/dataset:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ aidk-tensorflow /bin/bash -c "SIGOPT_API_TOKEN=${SIGOPT_API_TOKEN} USE_SIGOPT=${USE_SIGOPT} . /home/vmagent/app/hydro.ai/tests/cicd/jenkins_dien_test.sh"
```
For WnD:
```
$ docker run --rm --privileged --network host --device=/dev/dri -v /root/cicd_logs:/home/vmagent/app/cicd_logs -v /mnt/DP_disk1/dataset:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ aidk-tensorflow /bin/bash -c "SIGOPT_API_TOKEN=${SIGOPT_API_TOKEN} USE_SIGOPT=${USE_SIGOPT} . /home/vmagent/app/hydro.ai/tests/cicd/jenkins_wnd_test.sh"
```
For DLRM:
```
$ docker run --rm --privileged --network host --device=/dev/dri -v /root/cicd_logs:/home/vmagent/app/cicd_logs -v /mnt/DP_disk1/dataset:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ aidk-pytorch /bin/bash -c "SIGOPT_API_TOKEN=${SIGOPT_API_TOKEN} USE_SIGOPT=${USE_SIGOPT} . /home/vmagent/app/hydro.ai/tests/cicd/jenkins_dlrm_test.sh"
```
For RNNT:
```
docker run --rm --privileged --network host --device=/dev/dri -v /root/cicd_logs:/home/vmagent/app/cicd_logs -v /mnt/DP_disk1/dataset:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ aidk-pytorch110 /bin/bash -c "SIGOPT_API_TOKEN=${SIGOPT_API_TOKEN} USE_SIGOPT=${USE_SIGOPT} . /home/vmagent/app/hydro.ai/tests/cicd/jenkins_rnnt_test.sh"
```
For BERT:
```
docker run --rm --privileged --network host --device=/dev/dri -v /root/cicd_logs:/home/vmagent/app/cicd_logs -v /mnt/DP_disk1/dataset:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ aidk-tensorflow /bin/bash -c "SIGOPT_API_TOKEN=${SIGOPT_API_TOKEN} USE_SIGOPT=${USE_SIGOPT} . /home/vmagent/app/hydro.ai/tests/cicd/jenkins_bert_test.sh"
```
