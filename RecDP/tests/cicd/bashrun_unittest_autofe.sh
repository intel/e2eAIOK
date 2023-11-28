#!/bin/bash

failed_tests=""
echo "Setup pyrecdp latest package"
pip install -e .[autofe]

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

echo "test_feature_wrangler.TestFeatureWranglerPandasBased.test_nyc_taxi"
python -m unittest tests.test_feature_wrangler.TestFeatureWranglerPandasBased.test_nyc_taxi
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_wrangler.TestFeatureWranglerPandasBased.test_nyc_taxi\n"
fi

echo "test_feature_wrangler.TestFeatureWranglerPandasBased.test_twitter_recsys"
python -m unittest tests.test_feature_wrangler.TestFeatureWranglerPandasBased.test_twitter_recsys
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_wrangler.TestFeatureWranglerPandasBased.test_twitter_recsys\n"
fi

echo "test_feature_wrangler.TestFeatureWranglerPandasBased.test_amazon"
python -m unittest tests.test_feature_wrangler.TestFeatureWranglerPandasBased.test_amazon
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_wrangler.TestFeatureWranglerPandasBased.test_amazon\n"
fi

echo "test_feature_wrangler.TestFeatureWranglerPandasBased.test_frauddetect"
python -m unittest tests.test_feature_wrangler.TestFeatureWranglerPandasBased.test_frauddetect
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_wrangler.TestFeatureWranglerPandasBased.test_frauddetect\n"
fi

echo "test_pipeline_json.TestPipielineJson.test_import_nyc"
python -m unittest tests.test_pipeline_json.TestPipielineJson.test_import_nyc
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_pipeline_json.TestPipielineJson.test_import_nyc\n"
fi

echo "test_pipeline_json.TestPipielineJson.test_import_amazon"
python -m unittest tests.test_pipeline_json.TestPipielineJson.test_import_amazon
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_pipeline_json.TestPipielineJson.test_import_amazon\n"
fi

echo "test_pipeline_json.TestPipielineJson.test_import_twitter"
python -m unittest tests.test_pipeline_json.TestPipielineJson.test_import_twitter
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_pipeline_json.TestPipielineJson.test_import_twitter\n"
fi

echo "test_feature_wrangler.TestFeatureWranglerSparkBased.test_nyc_taxi"
python -m unittest tests.test_feature_wrangler.TestFeatureWranglerSparkBased.test_nyc_taxi
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_wrangler.TestFeatureWranglerSparkBased.test_nyc_taxi\n"
fi

echo "test_feature_wrangler.TestFeatureWranglerSparkBased.test_twitter_recsys"
python -m unittest tests.test_feature_wrangler.TestFeatureWranglerSparkBased.test_twitter_recsys
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_wrangler.TestFeatureWranglerSparkBased.test_twitter_recsys\n"
fi

echo "test_feature_wrangler.TestFeatureWranglerSparkBased.test_amazon"
python -m unittest tests.test_feature_wrangler.TestFeatureWranglerSparkBased.test_amazon
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_wrangler.TestFeatureWranglerSparkBased.test_amazon\n"
fi

echo "test_feature_wrangler.TestFeatureWranglerSparkBased.test_frauddetect"
python -m unittest tests.test_feature_wrangler.TestFeatureWranglerSparkBased.test_frauddetect
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_feature_wrangler.TestFeatureWranglerSparkBased.test_frauddetect\n"
fi

echo "test_pipeline_json.TestPipielineJson.test_import_nyc_execute_spark"
python -m unittest tests.test_pipeline_json.TestPipielineJson.test_import_nyc_execute_spark
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_pipeline_json.TestPipielineJson.test_import_nyc_execute_spark\n"
fi

echo "test_pipeline_json.TestPipielineJson.test_import_amazon_execute_spark"
python -m unittest tests.test_pipeline_json.TestPipielineJson.test_import_amazon_execute_spark
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_pipeline_json.TestPipielineJson.test_import_amazon_execute_spark\n"
fi

echo "test_pipeline_json.TestPipielineJson.test_import_twitter_execute_spark"
python -m unittest tests.test_pipeline_json.TestPipielineJson.test_import_twitter_execute_spark
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_pipeline_json.TestPipielineJson.test_import_twitter_execute_spark\n"
fi

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

echo "test_relational_builder.TestRelationalBuilder.test_outbrain"
python -m unittest tests.test_relational_builder.TestRelationalBuilder.test_outbrain
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_relational_builder.TestRelationalBuilder.test_outbrain\n"
fi

echo "test_relational_builder.TestRelationalBuilder.test_outbrain_spark"
python -m unittest tests.test_relational_builder.TestRelationalBuilder.test_outbrain_path
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_relational_builder.TestRelationalBuilder.test_outbrain_spark\n"
fi

echo "test_individual_method.TestUnitMethod.test_categorify"
python -m unittest tests.test_individual_method.TestUnitMethod.test_categorify
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_individual_method.TestUnitMethod.test_categorify\n"
fi

echo "test_individual_method.TestUnitMethod.test_group_categorify"
python -m unittest tests.test_individual_method.TestUnitMethod.test_group_categorify
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_individual_method.TestUnitMethod.test_group_categorify\n"
fi

echo "test_individual_method.TestUnitMethod.test_TE"
python -m unittest tests.test_individual_method.TestUnitMethod.test_TE
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_individual_method.TestUnitMethod.test_TE\n"
fi

echo "test_individual_method.TestUnitMethod.test_CE"
python -m unittest tests.test_individual_method.TestUnitMethod.test_CE
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_individual_method.TestUnitMethod.test_CE\n"
fi

if [ -z ${failed_tests} ]; then
    echo "All tests are passed"
else
    echo "*** Failed Tests are: ***"
    echo -e ${failed_tests}
    exit 1
fi
