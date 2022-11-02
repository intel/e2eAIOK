#!/usr/bin/env bash
set -e

release_version=$(cat version | head -1)
echo ${release_version} > dev/nbversion

python setup.py sdist
# e2eAIOK-0.2.0.tar.gz
twine check dist/e2eAIOK-${release_version}.tar.gz
twine upload dist/e2eAIOK-${release_version}.tar.gz