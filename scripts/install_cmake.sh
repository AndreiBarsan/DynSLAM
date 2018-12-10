#!/usr/bin/env bash

# TODO(andrei): Version check.

CPU_COUNT=$(grep -c ^processor /proc/cpuinfo)
CPU_COUNT=$(nproc)
CMAKE_VER="3.10.0"

set -eu

cd /tmp/

mkdir -p cmake
cd cmake
wget --no-check-certificate https://github.com/Kitware/CMake/archive/v${CMAKE_VER}.tar.gz || exit 1
tar xf v${CMAKE_VER}.tar.gz >/dev/null
# cd CMake-3.2.2
cd CMake-$CMAKE_VER

echo "Configuring CMake $CMAKE_VER..."
if [[ "$1" == "sudo" ]]; then
  echo "Configuring for system-wide installation"
  ./configure >/dev/null || exit 3
else
  echo "Configuring for local installation..."
  ./configure --prefix=~/.local || exit 3
fi

echo "Building CMake"
make -j$CPU_COUNT || exit 4

echo "Installing CMake"
make install || exit 5

echo "Cmake installed OK"


