#!/bin/bash
failed_tests=""

echo "Setup pyrecdp latest package"
python setup.py sdist && pip install dist/pyrecdp-*.*.*.tar.gz

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
    echo ${failed_tests}
    exit 1
fi
