#!/bin/bash

failed_tests=""
echo "Setup pyrecdp latest package"
python setup.py sdist && pip install dist/pyrecdp-*.*.*.tar.gz

echo "test_llmutils.Test_LLMUtils.test_near_dedup"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_near_dedup
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_near_dedup\n"
fi

echo "test_llmutils.Test_LLMUtils.test_near_dedup_spark"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_near_dedup_spark
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_near_dedup_spark\n"
fi

echo "test_llmutils.Test_LLMUtils.test_shrink_jsonl"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_shrink_jsonl
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_shrink_jsonl\n"
fi

echo "test_llmutils.Test_LLMUtils.test_text_to_jsonl"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_text_to_jsonl
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_text_to_jsonl\n"
fi

echo "test_llmutils.Test_LLMUtils.test_global_hash_jsonl"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_global_hash_jsonl
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_global_hash_jsonl\n"
fi

echo "test_llmutils.Test_LLMUtils.test_global_hash_parquet"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_global_hash_parquet
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_global_hash_parquet\n"
fi

echo "test_llmutils.Test_LLMUtils.test_get_hash_indexing"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_get_hash_indexing
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_get_hash_indexing\n"
fi

echo "test_llmutils.Test_LLMUtils.test_combine_hash_indexing"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_combine_hash_indexing
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_combine_hash_indexing\n"
fi

echo "test_llmutils.Test_LLMUtils.test_get_duplication_list"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_get_duplication_list
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_get_duplication_list\n"
fi

echo "test_llmutils.Test_LLMUtils.test_index_based_reduction"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_index_based_reduction
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_index_based_reduction\n"
fi

echo "test_llmutils.Test_LLMUtils.test_global_dedup"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_global_dedup
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_global_dedup\n"
fi

echo "test_llmutils.Test_LLMUtils.test_global_dedup_spark"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_global_dedup_spark
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_global_dedup_spark\n"
fi

echo "test_llmutils.Test_LLMUtils.test_filter_jsonl"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_filter_jsonl
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_filter_jsonl\n"
fi

echo "test_llmutils.Test_LLMUtils.test_bad_words_filter"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_bad_words_filter
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_bad_words_filter\n"
fi

echo "test_llmutils.Test_LLMUtils.test_length_filter"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_length_filter
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_length_filter\n"
fi

echo "test_llmutils.Test_LLMUtils.test_profanity_filter"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_profanity_filter
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_profanity_filter\n"
fi

echo "test_llmutils.Test_LLMUtils.test_text_fixer"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_text_fixer
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_text_fixer\n"
fi

echo "test_llmutils.Test_LLMUtils.test_pii_remove"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_pii_remove
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_pii_remove\n"
fi

echo "test_llmutils.Test_LLMUtils.test_language_identify"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_language_identify
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_language_identify\n"
fi

echo "test_llmutils.Test_LLMUtils.test_language_identify_spark"
python -m unittest tests.test_llmutils.Test_LLMUtils.test_language_identify_spark
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils.Test_LLMUtils.test_language_identify_spark\n"
fi

if [ -z ${failed_tests} ]; then
    echo "All tests are passed"
else
    echo "*** Failed Tests are: ***"
    echo ${failed_tests}
    exit 1
fi