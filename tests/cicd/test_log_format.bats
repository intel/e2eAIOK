#!/bin/bash

setup() {
    load 'test_helper/bats-support/load'
    load 'test_helper/bats-assert/load'
    load 'test_helper/bats-file/load'
    DIR="$( cd "$( dirname "$BATS_TEST_FILENAME" )" >/dev/null 2>&1 && pwd )"
    PATH="$DIR/src:$PATH"
}

@test "Check AIDK CI/CD Log Format" {
    run aidk_log_format.sh
    assert_output --partial 'Log pattern passed!'
}
