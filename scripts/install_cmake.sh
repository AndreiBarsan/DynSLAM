#!/usr/bin/env bash

# TODO(andrei): Version check.

CPU_COUNT=$(grep -c ^processor /proc/cpuinfo)
CPU_COUNT=${CPU_COUNT:-8}

set -eu

cd ~/work/DynSLAM

mkdir -p deps
cd deps
wget http://www.cmake.org/files/v3.2/cmake-3.2.2.tar.gz || exit 1
tar xf cmake-3.2.2.tar.gz >/dev/null
cd cmake-3.2.2

echo "Configuring CMake 3.2.2..."
./configure --prefix=~/.local >/dev/null || exit 3

echo "Building CMake 3.2.2..."
make -j$CPU_COUNT || exit 4

echo "Installing CMake 3.2.2..."
make install || exit 5

echo "Cmake installed OK"


