#!/bin/bash

failed_tests=""
echo "Setup pyrecdp latest package"
pip install -e .[LLM]

# to_test="test_quality_classifier test_quality_classifier_spark test_diversity_analysis test_near_dedup_spark test_global_dedup_spark test_pii_remove_spark test_pii_remove_email_spark test_pii_remove_phone_spark test_pii_remove_name_spark test_pii_remove_password_spark test_pii_remove_ip_spark test_pii_remove_key_spark test_sentence_split_spark test_text_pipeline_optimize_with_one_config_file test_text_pipeline_optimize_with_separate_config_file test_language_identify_spark"
# for test_name in `echo ${to_test}`; do
#     cmdline="python -m unittest tests.test_llmutils.Test_LLMUtils."${test_name}
#     echo "***************"
#     echo $cmdline
#     echo "***************"
#     ${cmdline}
#     if [ $? != 0 ]; then
#         failed_tests=${failed_tests}${cmdline}"\n"
#     fi
# done

# if [ -z ${failed_tests} ]; then
#     echo "All tests are passed"
# else
#     echo "*** Failed Tests are: ***"
#     echo -e ${failed_tests}
#     exit 1
# fi

python -m unittest tests.test_llmutils.Test_LLMUtils