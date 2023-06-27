#!/bin/bash

failed_tests=""
echo "Setup pyrecdp latest package"
python setup.py sdist && pip install dist/pyrecdp-*.*.*.tar.gz

echo "test_autofe.TestFE.test_nyc_taxi"
python -m unittest tests.test_autofe.TestFE.test_nyc_taxi
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_autofe.TestFE.test_nyc_taxi\n"
fi

echo "test_autofe.TestFE.test_frauddetect"
python -m unittest tests.test_autofe.TestFE.test_fraud_detect
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_autofe.TestFE.test_frauddetect\n"
fi

if [ -z ${failed_tests} ]; then
    echo "All tests are passed"
else
    echo "*** Failed Tests are: ***"
    echo ${failed_tests}
    exit 1
fi