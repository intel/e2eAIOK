#!/bin/bash

failed_tests=""

echo "Setup pyrecdp latest package"
python setup.py sdist && pip install dist/pyrecdp-*.*.*.tar.gz


echo "test_relational_builder.TestRelationalBuilder.test_outbrain"
python -m unittest tests.test_relational_builder.TestRelationalBuilder.test_outbrain
if [ $? != 0 ]; then
    failed_tests=${failed_tests}"test_relational_builder.TestRelationalBuilder.test_outbrain\n"
fi

if [ -z ${failed_tests} ]; then
    echo "All tests are passed"
else
    echo "*** Failed Tests are: ***"
    echo ${failed_tests}
fi