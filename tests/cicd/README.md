# AIDK CICD Integration Tests and Unit Tests

AIDK Integration Tests(Via BATS framework) provide end-to-end testing of AIDK and built-in workflows, which simulates the real world usage scenarios of AIDK.\
AIDK Unit Tests(Via Pytest framework) verify the code is working as expected by artificially created data of (input,expectation) pair.

## Prepare dataset

Internal copy sr602:/mnt/DP_disk1/dataset

## Test scripts that can be run manually by developer

The easiest way to run integration tests is with Docker.

Firstly, build docker for test.
```
$ cd ${AIDK_codebase}/tests/cicd/
$ docker build -t ${DOCKER_USERNAME}/oneapi-aikit:hydro.ai -f Dockerfile .
$ docker build -t ${DOCKER_USERNAME}/oneapi-aikit:legacy_hydro.ai -f DockerfileLegacy .
```

Then, run test script for specific workflow.\
For pipeline_test:
```
$ docker run --rm --privileged --network host --device=/dev/dri -v /root/cicd_logs:/home/vmagent/app/cicd_logs -v /mnt/DP_disk1/dataset:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ ${DOCKER_USERNAME}/oneapi-aikit:hydro.ai /bin/bash -c "SIGOPT_API_TOKEN=${SIGOPT_API_TOKEN} bash /home/vmagent/app/hydro.ai/tests/cicd/jenkins_pipeline_test.sh"
```
For DIEN:
```
$ docker run --rm --privileged --network host --device=/dev/dri -v /root/cicd_logs:/home/vmagent/app/cicd_logs -v /mnt/DP_disk1/dataset:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ ${DOCKER_USERNAME}/oneapi-aikit:hydro.ai /bin/bash -c "SIGOPT_API_TOKEN=${SIGOPT_API_TOKEN} bash /home/vmagent/app/hydro.ai/tests/cicd/jenkins_dien_test.sh"
```
For WnD:
```
$ docker run --rm --privileged --network host --device=/dev/dri -v /root/cicd_logs:/home/vmagent/app/cicd_logs -v /mnt/DP_disk1/dataset:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ ${DOCKER_USERNAME}/oneapi-aikit:hydro.ai /bin/bash -c "SIGOPT_API_TOKEN=${SIGOPT_API_TOKEN} bash /home/vmagent/app/hydro.ai/tests/cicd/jenkins_wnd_test.sh"
```
For DLRM:
```
$ docker run --rm --entrypoint "" --privileged --network host --device=/dev/dri -v /root/cicd_logs:/home/vmagent/app/cicd_logs -v /mnt/DP_disk1/dataset:/home/vmagent/app/dataset -v ${AIDK_codebase}:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ ${DOCKER_USERNAME}/oneapi-aikit:legacy_hydro.ai /bin/bash -c "SIGOPT_API_TOKEN=${SIGOPT_API_TOKEN} bash /home/vmagent/app/hydro.ai/tests/cicd/jenkins_dlrm_test.sh"
```
