#!/usr/bin/env bash
set -e

release_version=$(cat e2eAIOK/version | head -1)

python3 setup.py sdist
twine check dist/e2eAIOK-${release_version}.tar.gz
twine upload dist/e2eAIOK-${release_version}.tar.gz