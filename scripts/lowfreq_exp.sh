#!/usr/bin/env bash
# Execute this from the project root.
# Experiment for evaluating the impact on the static map quality of only doing the
# semantic segmentation and fusion every k frames, for k in {1, 2, 3 ... }.

if  [[ ! -d 'cmake-build-debug' || ! -d 'fig' || ! -d 'csv' || ! -d 'src' ]]; then
    echo >&2 "Please run this from the project root after building everything."
    exit 1
fi

set -u
IFS=$'\n\t'

cd cmake-build-debug && make DynSLAMGUI -j8 || exit 1

# ELAS results computed for K=(1, 2) only, so far.

#KS=(1 2 3 4 5) # 6 7 8 9 10)
#KS=(7 9 11 12 13 14 15 20 25)
KS=(1)
COUNT=${#KS[*]}

USE_DISPNETS=(true)

FRAME_LIMIT=1150
MIN_DECAY_AGE=300
MAX_DECAY_WEIGHT=99999
EVAL_DELAY=100

# TODO: rerun for k=4 with dispnet.

for USE_DISPNET in ${USE_DISPNETS[*]}; do
    for (( i = 0; i < $COUNT; i++ )); do
        K=${KS[i]}
        echo "Fuse every $K frames..."
        echo "Dispnet: $USE_DISPNET"
        echo
        echo

        cmd="./DynSLAMGUI \
            --dataset_root=/home/barsana/datasets/kitti/odometry-dataset/sequences/09 \
            --dynamic_mode=true                     \
            --enable_evaluation=true                \
            --min_decay_age=$MIN_DECAY_AGE          \
            --max_decay_weight=$MAX_DECAY_WEIGHT    \
            --evaluation_delay=$EVAL_DELAY          \
            --use_dispnet=${USE_DISPNET}            \
            --voxel_decay=true                      \
            --use_depth_weighting=true              \
            --frame_limit=$FRAME_LIMIT              \
            --fusion_every="$K"                     \
            --close_on_complete=true"

         echo "Command:"
         echo "$cmd"
         echo
         echo

         eval $cmd
    done
done
