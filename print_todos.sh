#!/usr/bin/env bash

ag '(TODO|TOOD|FIXME|XXX)' src --before=1 --after=3 --ignore-case "$@"
