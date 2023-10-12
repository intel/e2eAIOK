#!/bin/bash

failed_tests=""
echo "Setup pyrecdp latest package"
python setup.py sdist && pip install dist/pyrecdp-*.*.*.tar.gz

python -m unittest tests.test_llmutils_operations.Test_LLMUtils_Operations

# echo "test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextGlobalDeduplicate"
# python -m unittest tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextGlobalDeduplicate
# if [ $? != 0 ]; then
#     failed_tests=${failed_tests}"tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextGlobalDeduplicate\n"
# fi



# if [ -z ${failed_tests} ]; then
#     echo "All tests are passed"
# else
#     echo "*** Failed Tests are: ***"
#     echo -e ${failed_tests}
#     exit 1
# fi
