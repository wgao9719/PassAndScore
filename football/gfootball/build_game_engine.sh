#!/bin/bash
# Copyright 2019 Google LLC
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

set -e

LIB_EXTENSION="so"

if [[ "$OSTYPE" == "darwin"* ]] ; then
    LIB_EXTENSION="dylib"
fi

# Take into account # of cores and available RAM for deciding on compilation parallelism.
# We try to use psutil, if not available (or python3 failed) we fall back to 1 thread.
if ! PARALLELISM=$(python3 -c 'import psutil; import multiprocessing as mp; print(int(max(1,min((psutil.virtual_memory().available/1000000000-1)/0.5, mp.cpu_count()))))' 2>/dev/null); then
  PARALLELISM=1
fi

# Delete pre-existing version of CMakeCache.txt to make 'python3 -m pip install' work.
rm -f third_party/gfootball_engine/CMakeCache.txt
PY_EXEC=$(python3 -c "import sys; print(sys.executable)")
echo "Using Python executable: $PY_EXEC"
pushd third_party/gfootball_engine && cmake . -DPython_EXECUTABLE="$PY_EXEC" -DPython3_EXECUTABLE="$PY_EXEC" -DBoost_NO_BOOST_CMAKE=ON && make -j $PARALLELISM && popd
pushd third_party/gfootball_engine && ln -sf libgame.$LIB_EXTENSION _gameplayfootball.so && popd
