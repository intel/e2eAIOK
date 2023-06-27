#!/bin/bash
failed_tests=""

echo "Setup pyrecdp latest package"
python setup.py sdist && pip install dist/pyrecdp-*.*.*.tar.gz

echo "test_feature_profiler.TestFeatureProfiler.test_nyc_taxi"
python -m unittest tests.test_feature_profiler.TestFeatureProfiler.test_nyc_taxi
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_profiler.TestFeatureProfiler.test_nyc_taxi\n"
fi

echo "test_feature_profiler.TestFeatureProfiler.test_twitter_recsys"
python -m unittest tests.test_feature_profiler.TestFeatureProfiler.test_twitter_recsys
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_profiler.TestFeatureProfiler.test_twitter_recsys\n"
fi

echo "test_feature_profiler.TestFeatureProfiler.test_amazon"
python -m unittest tests.test_feature_profiler.TestFeatureProfiler.test_amazon
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_profiler.TestFeatureProfiler.test_amazon\n"
fi

echo "test_feature_profiler.TestFeatureProfiler.test_frauddetect"
python -m unittest tests.test_feature_profiler.TestFeatureProfiler.test_frauddetect
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_profiler.TestFeatureProfiler.test_frauddetect\n"
fi

if [ -z ${failed_tests} ]; then
    echo "All tests are passed"
else
    echo "*** Failed Tests are: ***"
    echo ${failed_tests}
fi