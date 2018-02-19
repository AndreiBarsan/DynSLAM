#!/usr/bin/env bash

# TODO(andrei): Version check.

CPU_COUNT=$(grep -c ^processor /proc/cpuinfo)
CPU_COUNT=${CPU_COUNT:-8}

set -eu

cd /tmp/

mkdir -p cmake
cd cmake
wget --no-check-certificate https://github.com/Kitware/CMake/archive/v3.2.2.tar.gz || exit 1
tar xf v3.2.2.tar.gz >/dev/null
cd CMake-3.2.2

echo "Configuring CMake 3.2.2..."
if [[ "$1" == "sudo" ]]; then
    ./configure >/dev/null || exit 3
else
    ./configure --prefix=~/.local >/dev/null || exit 3
fi

echo "Building CMake 3.2.2..."
make -j$CPU_COUNT || exit 4

echo "Installing CMake 3.2.2..."
make install || exit 5

echo "Cmake installed OK"


