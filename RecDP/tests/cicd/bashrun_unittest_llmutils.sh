#!/bin/bash

failed_tests=""
echo "Setup pyrecdp latest package"
python setup.py sdist && pip install dist/pyrecdp-*.*.*.tar.gz

# call cmdline tests
cmdline="python pyrecdp/primitives/llmutils/quality_classifier.py --dataset_path tests/data/llm_data/arxiv_sample_100.jsonl --result_path tests/data/output/qualify_classify"
echo "***************"
echo $cmdline
echo "***************"
${cmdline}
if [ $? != 0 ]; then
    failed_tests=${failed_tests}${cmdline}"\n"
fi

# cmdline="python pyrecdp/primitives/llmutils/global_dedup.py -d tests/data/PILE/ -o tests/data/PILE_global_dedup -t jsonl"
# echo "***************"
# echo $cmdline
# echo "***************"
# ${cmdline}
# if [ $? != 0 ]; then
#     failed_tests=${failed_tests}${cmdline}"\n"
# fi

# cmdline="python pyrecdp/primitives/llmutils/near_dedup.py -d tests/data/PILE/ -o tests/data/PILE_near_dedup -t jsonl"
# echo "***************"
# echo $cmdline
# echo "***************"
# ${cmdline}
# if [ $? != 0 ]; then
#     failed_tests=${failed_tests}${cmdline}"\n"
# fi

# if [ -z ${failed_tests} ]; then
#     echo "All tests are passed"
# else
#     echo "*** Failed Tests are: ***"
#     echo -e ${failed_tests}
#     exit 1
# fi

python -m unittest tests.test_llmutils.Test_LLMUtils