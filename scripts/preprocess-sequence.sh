#!/usr/bin/env bash
# Preprocesses a KITTI sequence for DynSLAM.
# And yes, the run time estimates from the paper DO include this time!
#
# Usage:
#   ./preprocess-sequence.sh <type> <dataset-root> training/testing <sequence-id>
#
# Example (if your KITTI tracking data was downloaded in data/kittti/tracking):
#   ./preprocess-sequence.sh kitti-tracking ./data/kitti/tracking training 0

# TODO XXX under heavy construction atm
# maybe it would be better to just implement this in Python?

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ORIGINAL_WD="$(pwd)"
source "$SCRIPT_DIR/utils.sh.inc"

# TODO proper arg validation

DATASET_TYPE="$1"
DATASET_ROOT="$2"
SEQUENCE_ID="$3"

fail_tracking_structure () {
    good_tree=""
    read -r -d '' good_tree <<- EOM
        ├── training
        │   ├── calib
        │   ├── image_02
        │   │   ├── 0000
        │   │   └── etc.
        │   ├── image_03
        │   │   ├── 0000
        │   │   └── etc.
        │   └── velodyne
        │       ├── 0000
        │       └── etc.
        └── testing
            └── <same as training>
EOM
    fail "Invalid directory structure for the tracking dataset. It should be: $good_tree."
}

ensure_tracking_ok () {
    # Ensures the KITTI tracking dataset is downloaded and valid.
    ds_root="$1"
    seq_id="$2" # TODO(andreib): Use this.

    cd "$ds_root"
    if [[ -d 'training' ]] && [[ -d 'testing' ]]; then
        # Canary-style check. If these folders are in place, we can assume the
        # dataset is probably OK.
        cd 'training'
        if ! [[ -d 'calib' && -d 'image_02' && -d 'image_03' && -d 'velodyne' ]]; then
            fail_tracking_structure
        fi
        cd '../testing'
        if ! [[ -d 'calib' && -d 'image_02' && -d 'image_03' && -d 'velodyne' ]]; then
            fail_tracking_structure
        fi
    else
        fail_tracking_structure
    fi
}

ensure_odometry_ok () {
    # Ensures the KITTI odometry dataset is downloaded and valid.
    # TODO(andreib): Implement.
    true
}

ensure_sequence_ok () {
    ds_type="$1"
    ds_root="$2"
    seq_id="$3"
    shift
    if [[ "$ds_type" == "kitti-tracking" ]]; then
        ensure_tracking_ok "$ds_root" "$seq_id"
    elif [[ "$ds_type" == "kitti-odometry" ]]; then
        ensure_odometry_ok "$ds_root" "$seq_id"
    else
        fail "Unknown dataset type: [$ds_type]"
    fi
}

prepare_depth_dispnet () {
    ds_type="$1"
    ds_root="$2"
    # TODO(andreib): Support this generically.
    sequence_type="testing"
    sequence_id="$3"

    cd "$SCRIPT_DIR/.."

    if ! [[ -d 'preprocessing/dispnet-flownet-docker' ]]; then
        echo "Could not find DispNet folder. Did you clone this repository recursively?"
        while true; do
            read -p "Do you wish to fetch the git submodules now? [yes/no] " yn
            case $yn in
                [Yy]* ) git submodule update --init --recursive; break;;
                [Nn]* ) fail "DispNet not found. Preprocessing cancelled."; break ;;
                *)      echo "Please enter yes or no."
            esac
        done
    fi

    path="$ds_root/$sequence_type"
    seq_depth_root="$path/precomputed-depth-dispnet/$(printf '%04d' $sequence_id)"
    if ! [[ -d  "$seq_depth_root" ]]; then
        mkdir -pv "$seq_depth_root"
    fi

    (
        cd preprocessing/dispnet-flownet-docker
        ls $ORIGINAL_WD/$ds_root
        # TODO(andreib): Support ds_root absolute...
        echo "Running dispnet proc script!"
        echo "$ORIGINAL_WD/$ds_root" is the root dir
        ./process-kitti-tracking.sh "$ORIGINAL_WD/$ds_root" "$sequence_type" "$sequence_id"
    ) || fail "Could not run DispNet on sequence ID [$sequence_id] under path: [$path]"
}

prepare_depth_elas () {
    ds_type="$1"
    ds_root="$2"

    # TODO(andreib): Implement.
    echo >&2 "Warning: Preprocessing with ELAS not supported yet!"
}

prepare_depth () {
    prepare_depth_dispnet "$@"
    prepare_depth_elas "$@"
}

prepare_semantic_mnc () {
    ds_type="$1"
    ds_root="$2"
    # TODO(andreib): Implement.
}

prepare_semantic () {
    ds_type="$1"
    ds_root="$2"
    prepare_semantic_mnc "$ds_type" "$ds_root"
}

#echo "Will operate from the DynSLAM project root [$SCRIPT_DIR/..]."
#cd "$SCRIPT_DIR/.."

ensure_sequence_ok "$DATASET_TYPE" "$DATASET_ROOT" "$SEQUENCE_ID"
prepare_depth "$DATASET_TYPE" "$DATASET_ROOT" "$SEQUENCE_ID"
prepare_semantic "$DATASET_TYPE" "$DATASET_ROOT" "$SEQUENCE_ID"
