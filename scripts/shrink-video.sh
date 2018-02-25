#!/usr/bin/env bash
# Script used to shrink our rendered demo video such that it fits within IEEE's
# 10Mb limit.

set -eu

INPUT="$1"
ffmpeg -i "$INPUT" -vf scale=1000:610 "rescaled-$INPUT"
