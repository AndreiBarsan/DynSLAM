#!/usr/bin/env bash
# Execute this from the project root.
# Runs basic experiments on the KITTI tracking sequences.

if  [[ ! -d 'cmake-build-debug' || ! -d 'fig' || ! -d 'csv' || ! -d 'src' ]]; then
    echo >&2 "Please run this from the project root after building everything."
    exit 1
fi

set -u
IFS=$'\n\t'

cd cmake-build-debug && make DynSLAMGUI -j8 || exit 1

TRACKING_ROOT=~/datasets/kitti/tracking-dataset/

# Sequences [0-8] was done on 24 August, evening.
# Sequence [1] ... pending (running when I left, dispnet+dynamic)
# SEQUENCES=(1 2 3 4 5 6 7 8 9)
#SEQUENCES=(8 9 10 11 12 13 14 15 16 17 18 19 20)
# These seem to be missing segmentation or depth data? (low-priority; focus on 0-9, i.e., first 10).
# (19 13)
SEQUENCES=(1)
USE_DISPNET_OPTIONS=(true false)
#DYNAMIC_MODE_OPTIONS=(false true)
DYNAMIC_MODE_OPTIONS=(true)

# Careful you don't mess up the ~280 mostly-still sequence 06!
MIN_DECAY_AGE=300
MAX_DECAY_WEIGHT=99999

SEQ_COUNT=${#SEQUENCES[*]}

touch command_log.txt
printf '\n\n\n\n' >> command_log.txt

# For these experiments, we only care about input vs. fused, since there's little traffic.
DYNAMIC_MODE=true

for (( i = 0; i < $SEQ_COUNT; i++ )); do
    for USE_DISPNET in ${USE_DISPNET_OPTIONS[*]}; do
        for DYNAMIC_MODE in ${DYNAMIC_MODE_OPTIONS[*]}; do
            SEQ_ID="${SEQUENCES[i]}"
            SEQ_LABEL=$(printf '%02d' $SEQ_ID)
            echo "Processing sequence $SEQ_LABEL"
            echo "USE_DISPNET=$USE_DISPNET"
            echo "DYNAMIC_MODE=$DYNAMIC_MODE"

            cmd="./DynSLAMGUI \
                --dataset_root=$TRACKING_ROOT
                --dataset_type=kitti-tracking           \
                --kitti_tracking_sequence_id=$SEQ_ID    \
                --dynamic_mode=$DYNAMIC_MODE            \
                --enable_evaluation=true                \
                --min_decay_age=$MIN_DECAY_AGE           \
                --max_decay_weight=$MAX_DECAY_WEIGHT     \
                --evaluation_delay=0                    \
                --use_dispnet=${USE_DISPNET}            \
                --voxel_decay=true                      \
                --use_depth_weighting=true              \
                --close_on_complete=true"
            echo "Command: $cmd"
            echo "$cmd" >> command_log.txt
            echo "Away we go.."
            echo
            eval $cmd
        done
    done
done
