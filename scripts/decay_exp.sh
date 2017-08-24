#!/usr/bin/env bash
# Execute this from the project root.

set -euo pipefail
IFS=$'\n\t'

cd cmake-build-debug && make DynSLAMGUI -j8 || exit 1

WEIGHTS=(1 2 3 5 8 10)
MIN_DECAY_AGES=(40 40 40 40 80 80 80)
EVAL_DELAYS=(80 80 80 80 120 120 120)
FRAME_LIMITS=(1085 1085 1085 1085 1125 1125 1125)
COUNT=${#WEIGHTS[*]}

USE_DISPNET="true"

#for weight in ${WEIGHTS[*]}; do
for (( i = 0; i < $COUNT; i++ )); do
    weight="${WEIGHTS[i]}"
    min_age="${MIN_DECAY_AGES[i]}"
    eval_delay="${EVAL_DELAYS[i]}"
    frame_limit="${FRAME_LIMITS[i]}"

    echo "Weight: $weight"
    echo "$min_age, $eval_delay, $frame_limit"
    echo
    echo

    # TODO(andrei): ulimit that shit to ~32gb of ram to ensure no thrashing.
    ./DynSLAMGUI \
    --dataset_root=/home/barsana/datasets/kitti/odometry-dataset/sequences/09 \
    --dynamic_mode=true             \
    --enable_evaluation=true         \
    --min_decay_age=$min_age               \
    --max_decay_weight=$weight      \
    --evaluation_delay=$eval_delay            \
    --use_dispnet=${USE_DISPNET}    \
    --voxel_decay=true              \
    --use_depth_weighting=false     \
    --frame_limit=$frame_limit
    --close_on_complete=true
done

