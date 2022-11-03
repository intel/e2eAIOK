#!/usr/bin/env bash
set -e

release_version=$(cat version | head -1)
echo ${release_version} > e2eAIOK/version

python3 setup.py sdist
twine check dist/e2eAIOK-${release_version}.tar.gz
twine upload dist/e2eAIOK-${release_version}.tar.gz
