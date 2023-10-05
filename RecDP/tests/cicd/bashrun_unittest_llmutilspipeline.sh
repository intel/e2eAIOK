#!/bin/bash

failed_tests=""
echo "Setup pyrecdp latest package"
python setup.py sdist && pip install dist/pyrecdp-*.*.*.tar.gz

echo "test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextNormalize"
python -m unittest tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextNormalize
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextNormalize\n"
fi

echo "test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextBytesize"
python -m unittest tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextBytesize
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextBytesize\n"
fi

echo "test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextSourceId"
python -m unittest tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextSourceId
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextSourceId\n"
fi

echo "test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextLengthFilter"
python -m unittest tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextLengthFilter
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextLengthFilter\n"
fi

echo "test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextBadwordsFilter"
python -m unittest tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextBadwordsFilter
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextBadwordsFilter\n"
fi

echo "test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextProfanityFilter"
python -m unittest tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextProfanityFilter
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextProfanityFilter\n"
fi

echo "test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextFixer"
python -m unittest tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextFixer
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextFixer\n"
fi

echo "test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextLanguageIdentify"
python -m unittest tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextLanguageIdentify
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextLanguageIdentify\n"
fi

echo "test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextDocumentSplit"
python -m unittest tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextDocumentSplit
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextDocumentSplit\n"
fi

echo "test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextPIIRemoval"
python -m unittest tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextPIIRemoval
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextPIIRemoval\n"
fi

echo "test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextURLFilter"
python -m unittest tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextURLFilter
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextURLFilter\n"
fi

echo "test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextFuzzyDeduplicate"
python -m unittest tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextFuzzyDeduplicate
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextFuzzyDeduplicate\n"
fi

echo "test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextGlobalDeduplicate"
python -m unittest tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextGlobalDeduplicate
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"tests.test_llmutils_pipeline.Test_LLMUtils_Pipeline.test_TextGlobalDeduplicate\n"
fi



if [ -z ${failed_tests} ]; then
    echo "All tests are passed"
else
    echo "*** Failed Tests are: ***"
    echo -e ${failed_tests}
    exit 1
fi
