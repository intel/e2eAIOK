# AIDK CICD Integration Tests and Unit Tests

AIDK Integration Tests(Via BATS framework) provide end-to-end testing of AIDK and built-in workflows(pipeline_test/DIEN/WnD/DLRM), which simulate the real world usage scenarios of AIDK.\
AIDK Unit Tests(Via Pytest framework) verify the code is working as expected by artificially created data of (input,expectation) pair.


## How to create a test script for your workload

Create a test script named jenkins_${model}_test.sh under tests/cicd folder, and then define your model specific MODEL_NAME, DATA_PATH, CONF_FILE.

## Prepare dataset

Internal copy sr602:/mnt/DP_disk1/dataset

## Test scripts that can be run manually by developer

The easiest way to run test scripts is with Docker.

Firstly, build docker for test.
```
$ cd ${AIDK_codebase}/tests/cicd/
$ docker build -t test-aidk -f Dockerfile .
$ docker build -t legacy-test-aidk -f DockerfileLegacy .
```

Then, run test script for specific workflow.\
For pipeline_test:
```
$ docker run --rm --privileged --network host --device=/dev/dri -v /root/cicd_logs:/home/vmagent/app/cicd_logs -v /mnt/DP_disk1/dataset:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ test-aidk /bin/bash -c "SIGOPT_API_TOKEN=${SIGOPT_API_TOKEN} USE_SIGOPT=0 bash /home/vmagent/app/hydro.ai/tests/cicd/jenkins_pipeline_test.sh"
```
For DIEN:
```
$ docker run --rm --privileged --network host --device=/dev/dri -v /root/cicd_logs:/home/vmagent/app/cicd_logs -v /mnt/DP_disk1/dataset:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ test-aidk /bin/bash -c "SIGOPT_API_TOKEN=${SIGOPT_API_TOKEN} USE_SIGOPT=0 bash /home/vmagent/app/hydro.ai/tests/cicd/jenkins_dien_test.sh"
```
For WnD:
```
$ docker run --rm --privileged --network host --device=/dev/dri -v /root/cicd_logs:/home/vmagent/app/cicd_logs -v /mnt/DP_disk1/dataset:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ test-aidk /bin/bash -c "SIGOPT_API_TOKEN=${SIGOPT_API_TOKEN} USE_SIGOPT=0 bash /home/vmagent/app/hydro.ai/tests/cicd/jenkins_wnd_test.sh"
```
For DLRM:
```
$ docker run --rm --entrypoint "" --privileged --network host --device=/dev/dri -v /root/cicd_logs:/home/vmagent/app/cicd_logs -v /mnt/DP_disk1/dataset:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ legacy-test-aidk /bin/bash -c "SIGOPT_API_TOKEN=${SIGOPT_API_TOKEN} USE_SIGOPT=0 bash /home/vmagent/app/hydro.ai/tests/cicd/jenkins_dlrm_test.sh"
```
