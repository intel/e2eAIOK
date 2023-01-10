#!/bin/bash

setup() {
    load 'test_helper/bats-support/load'
    load 'test_helper/bats-assert/load'
    load 'test_helper/bats-file/load'
    DIR="$( cd "$( dirname "$BATS_TEST_FILENAME" )" >/dev/null 2>&1 && pwd )"
    PATH="$DIR/src:$PATH"
}

@test "Check e2eAIOK DE-NAS Best Structure" {
    assert_file_exist /home/vmagent/app/e2eaiok/e2eAIOK/DeNas/best_model_structure.txt
}
