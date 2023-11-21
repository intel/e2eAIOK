#!/bin/bash

failed_tests=""
echo "Setup pyrecdp latest package"
pip install -e .[autofe]

echo "test_spark_dataprocessor.TestSparkDataProcessor.test_local"
python -m unittest tests.test_spark_dataprocessor.TestSparkDataProcessor.test_local
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_spark_dataprocessor.TestSparkDataProcessor.test_local\n"
fi

echo "test_spark_dataprocessor.TestSparkDataProcessor.test_ray"
python -m unittest tests.test_spark_dataprocessor.TestSparkDataProcessor.test_ray
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_spark_dataprocessor.TestSparkDataProcessor.test_ray\n"
fi

if [ -z ${failed_tests} ]; then
    echo "All tests are passed"
else
    echo "*** Failed Tests are: ***"
    echo ${failed_tests}
    exit 1
fi