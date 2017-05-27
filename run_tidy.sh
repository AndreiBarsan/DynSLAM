#!/usr/bin/env bash
# Runs clang-tidy on the source tree. Requires 'clang-tidy' and cmake 3.6+.

set -euo pipefail
IFS=$'\n\t'

mkdir -p build-clang-tidy
cd $_

cmake -DCMAKE_CXX_CLANG_TIDY:STRING="clang-tidy;-checks=-*,readability-*" ..
make -j8 2>&1 | tee clang-tidy-report.txt

