#!/usr/bin/env bash
#
# Copyright      2022  Xiaomi Corp.       (author: Fangjun Kuang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The following environment variables are supposed to be set by users
#
# - LILCOM_CONDA_TOKEN
#     If not set, auto upload to anaconda.org is disabled.
#
#     Its value is from https://anaconda.org/k2-fsa/settings/access
#      (You need to login as user k2-fsa to see its value)
#
set -e
export CONDA_BUILD=1

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
lilcom_dir=$(cd $cur_dir/.. && pwd)

cd $lilcom_dir

export LILCOM_ROOT_DIR=$lilcom_dir
echo "LILCOM_DIR: $LILCOM_ROOT_DIR"

LILCOM_PYTHON_VERSION=$(python -c "import sys; print('.'.join(sys.version.split('.')[:2]))")

# Example value: 3.8
export LILCOM_PYTHON_VERSION

if [ -z $LILCOM_CONDA_TOKEN ]; then
  echo "Auto upload to anaconda.org is disabled since LILCOM_CONDA_TOKEN is not set"
  conda build --no-test --no-anaconda-upload ./scripts/conda/lilcom
else
  conda build --no-test --token $LILCOM_CONDA_TOKEN ./scripts/conda/lilcom
fi
