#!/bin/bash

echo "Performance Test will be setup later"
echo "Setup pyrecdp latest package"
pip install -e .[autofe]

echo "test_feature_wrangler.TestFeatureWranglerSparkBased.test_nyc_taxi_perf"
nyc_result=`python -m unittest tests.test_feature_wrangler.TestFeatureWranglerSparkBased.test_nyc_taxi_perf | grep "NYC taxi performance test took"`
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_wrangler.TestFeatureWranglerSparkBased.test_nyc_taxi_perf\n"
else
    results=${results}${nyc_result}"\n"
fi

echo "test_feature_wrangler.TestFeatureWranglerSparkBased.test_frauddetect_perf"
fd_result=`python -m unittest tests.test_feature_wrangler.TestFeatureWranglerSparkBased.test_frauddetect_perf | grep "Fraud detect performance test took"`
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_wrangler.TestFeatureWranglerSparkBased.test_frauddetect_perf\n"
else
    results=${results}${fd_result}"\n"
fi

echo "test_feature_wrangler.TestFeatureWranglerSparkBased.test_recsys2023_perf"
recsys_result=`python -m unittest tests.test_feature_wrangler.TestFeatureWranglerSparkBased.test_recsys2023_perf | grep "Recsys2023 performance test took"`
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_wrangler.TestFeatureWranglerSparkBased.test_recsys2023_perf\n"
else
    results=${results}${recsys_result}"\n"
fi

if [ -z ${failed_tests} ]; then
    echo "All tests are passed"
    echo ${results}
else
    echo "*** Failed Tests are: ***"
    echo ${failed_tests}
    echo ${results}
    exit 1
fi