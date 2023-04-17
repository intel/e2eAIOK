#    Copyright 2022, Intel Corporation.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

#!/bin/bash
set -x

current_dir=`pwd`

# clone code
git clone https://github.com/intel/e2eAIOK.git
cd e2eAIOK
git submodule update --init modelzoo/third_party/nnUNet

# unpack patch
cd modelzoo/unet && sh patch_unet.sh 

# remove useless code
cd $current_dir
mv e2eAIOK/modelzoo/unet .
rm -rf e2eAIOK