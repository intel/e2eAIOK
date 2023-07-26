#!/bin/bash

failed_tests=""
echo "Setup pyrecdp latest package"
python setup.py sdist && pip install dist/pyrecdp-*.*.*.tar.gz

echo "test_autofe.TestFE.test_nyc_taxi_pandas"
python -m unittest tests.test_autofe.TestFE.test_nyc_taxi_pandas
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_autofe.TestFE.test_nyc_taxi_pandas\n"
fi

echo "test_autofe.TestFE.test_fraud_detect_pandas"
python -m unittest tests.test_autofe.TestFE.test_fraud_detect_pandas
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_autofe.TestFE.test_fraud_detect_pandas\n"
fi

echo "test_autofe.TestFE.test_nyc_taxi_spark"
python -m unittest tests.test_autofe.TestFE.test_nyc_taxi_spark
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_autofe.TestFE.test_nyc_taxi_spark\n"
fi

echo "test_autofe.TestFE.test_fraud_detect_spark"
python -m unittest tests.test_autofe.TestFE.test_fraud_detect_spark
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_autofe.TestFE.test_fraud_detect_spark\n"
fi

if [ -z ${failed_tests} ]; then
    echo "All tests are passed"
else
    echo "*** Failed Tests are: ***"
    echo ${failed_tests}
    exit 1
fi