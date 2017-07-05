#!/usr/bin/env bash

ag --before=1 --after=3 --ignore-case "$@" '(TODO|TOOD|FIXME|XXX)' src/InfiniTAM src/DynSLAM
