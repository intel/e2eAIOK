#!/usr/bin/env bash

set -e

release_version=$(cat version | head -1)
nightly_build_date=`date '+%Y%m%d'`
nightly_version=${release_version}b${nightly_build_date}
echo $nightly_version > version

python setup.py bdist_wheel --python-tag py3
twine check dist/e2eAIOK-${nightly_version}-py3-none-any.whl
twine upload dist/e2eAIOK-${nightly_version}-py3-none-any.whl
