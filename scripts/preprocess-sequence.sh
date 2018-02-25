#!/usr/bin/env bash
# Preprocesses a KITTI sequence for DynSLAM.
# And yes, the run time estimates from the paper DO include this time!
#
# Usage:
#   ./preprocess-sequence.sh <type> <dataset-root> training/testing <sequence-id>
#
# Requirements:
#   * An NVIDIA GPU with at least 6Gb of memory (the MNC architecture is a
#    little memory-hungry)
#   * CUDA 8
#   * nvidia-docker


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ORIGINAL_WD="$(pwd)"
DYNSLAM_ROOT="$SCRIPT_DIR/.."
source "$SCRIPT_DIR/utils.sh.inc"

usage () {
    cat << EOF
Usage: $0 <type> <dataset-root> <split> <sequence-id>

Example (if your KITTI tracking data was downloaded in data/kittti/tracking):
   ./preprocess-sequence.sh kitti-tracking ./data/kitti/tracking training 0



EOF
    exit 1
}

if [[ "$#" -ne 4 ]]; then
    usage
fi

DATASET_TYPE="$1"
DATASET_ROOT="$2"
DATASET_SPLIT="$3"
SEQUENCE_ID="$4"

submodule_update_or_fail () {
    # Helper which prompts the user to grab the necessary git submodule, in
    # case they are missing.
    label="$1"
    echo "Could not find $label folder. Did you clone this repository recursively?"
    while true; do
        read -p "Do you wish to fetch the git submodules now? [yes/no] " yn
        case $yn in
            [Yy]* ) git submodule update --init --recursive; break;;
            [Nn]* ) fail "$label not found. Preprocessing cancelled."; break ;;
            *)      echo "Please enter yes or no."
        esac
    done
}

adjust_dataset_root () {
    ds_root="$1"

    if [[ "$ds_root" = /* ]]; then
        echo "$ds_root"
    else
        # Since we're working in a directory that's (likely) different from
        # the one the script was called from, ensure the dataset root path
        # is still correct.
        echo "$ORIGINAL_WD/$ds_root"
    fi
}

fail_tracking_structure () {
    # Called when the KITTI Tracking Dataset directory structure is incorrect.
    good_tree=""
    read -r -d '' good_tree <<EOM
    Expected directory structure:
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
    fail "Invalid directory structure for the tracking dataset. \n$good_tree."
}

fail_odometry_structure () {
    # Called when the KITTI Odometry Dataset directory structure is incorrect.
    # TODO(andreib): Print out the expected odometry dataset structure.
    fail "Not yet implemented."
}

ensure_tracking_ok () {
    # Ensures the KITTI tracking dataset is downloaded and valid.
    ds_root="$1"
    ds_split="$2"
    seq_id="$3"

    if ! [[ "$ds_split" == "training" || "$ds_split" == "testing" ]]; then
        fail "Invalid dataset split: [$ds_split]. Acceptable values are \"training\" and \"testing\"."
    fi

    cd "$ds_root"
    if [[ -d "$ds_split" ]] ; then
        # Canary-style check. If these folders are in place, we can assume the
        # dataset is probably OK.
        cd "$ds_split"
        seq_id_pad="$(printf '%04d' "$seq_id")"
        if ! [[ -f "calib/${seq_id_pad}.txt"    && \
                -d "image_02/$seq_id_pad" && \
                -d "image_03/$seq_id_pad" && \
                -d "velodyne/$seq_id_pad" ]]; then
            fail_tracking_structure
        fi
    else
        fail_tracking_structure
    fi
}

ensure_odometry_ok () {
    # Ensures the KITTI odometry dataset is downloaded and valid.
    # TODO(andreib): Implement.
    fail "Odometry sequence preprocessing not yet implemented (only difference from tracking is a different directory structure)."
}

ensure_sequence_ok () {
    ds_type="$1"
    shift

    if [[ "$ds_type" == "kitti-tracking" ]]; then
        ensure_tracking_ok "$@"
    elif [[ "$ds_type" == "kitti-odometry" ]]; then
        ensure_odometry_ok "$@"
    else
        fail "Unknown dataset type: [$ds_type]"
    fi
}

prepare_depth_dispnet () {
    # Precomputed dense depth maps from the stereo pairs in the given sequence
    # using the DispNet architecture. Leverages 'nvidia-docker' for isolation.
    ds_type="$1"
    ds_root="$2"
    ds_split="$3"
    seq_id="$4"

    cd "$DYNSLAM_ROOT"

    path="$ds_root/$ds_split"
    seq_depth_root="$path/precomputed-depth-dispnet/$(printf '%04d' $seq_id)"

    DISPNET_DIR="preprocessing/dispnet-flownet-docker"
    if ! [[ -d "$DISPNET_DIR" ]]; then
        submodule_update_or_fail "DispNet"
    fi

    if [[ -f "$seq_depth_root/000000.pfm" ]]; then
        # TODO-LOW(andreib): Check count, too, like you do for the semantics.
        echo "Found a generated .pfm file in the dir; assuming depth already computed."
        local lines=$(ls $seq_depth_root | wc -l )
        echo "$lines files"
        return
    fi

    (
        cd "$DISPNET_DIR"
        ds_root=$(adjust_dataset_root "$ds_root")
        echo "Running dispnet preprocessing script!"
        if [[ "$ds_type" == "kitti-tracking" ]]; then
            ./process-kitti-tracking.sh "$ds_root" "$ds_split" "$seq_id"
        elif [[ "$ds_type" == "kitti-odometry" ]]; then
            ./process-kitti.sh "$ds_root" "$ds_split" "$seq_id"
        else
            echo >&2 "Unknown dataset type [$ds_type] when attempting to preprocess depth using DispNet."
            exit 1
        fi
    ) || fail "Could not build or run DispNet on sequence ID [$seq_id] under path: [$path]"
}

prepare_depth_elas () {
    ds_type="$1"
    ds_root="$2"

    # TODO(andreib): Implement using your tool. Would just be yet another submodule. :)
    echo >&2 "Warning: Preprocessing with ELAS not supported yet!"
}

prepare_depth () {
    prepare_depth_dispnet "$@"
    prepare_depth_elas "$@"
}

prepare_semantic_mnc () {
    local ds_type="$1"
    local ds_root="$2"
    local ds_split="$3"
    local seq_id="$4"

    # TODO(andreib): Do we need all these checks and mkdir -p's here?
    cd "$DYNSLAM_ROOT"
    path="$ds_root/$ds_split"
    # TODO(andreib): Use functions to compute these paths so you can easily
    # support both KITTI tracking and odometry.
    seq_seg_root="$path/seg_image_02/$(printf '%04d' $seq_id)"
    seq_image_02_root="$path/image_02/$(printf '%04d' $seq_id)"

    local mnc_dir="preprocessing/MNC"
    if ! [[ -d "$mnc_dir" ]]; then
        submodule_update_or_fail "MNC"
    fi
    if [[ -d "$seq_seg_root" ]]; then
        local frame_count=$(ls $seq_image_02_root/*.png | wc -l)
        local segmented_count=$(ls $seq_seg_root/final_* | wc -l)
        if [[ "$frame_count" = "$segmented_count" ]]; then
            echo "Sequence [$seq_id] already segmented."
            return 0
        fi
    else
        mkdir -pv "$seq_seg_root"
    fi
    (cd "$mnc_dir" && nvidia-docker build -t mnc .) || fail "Could not build the MNC Docker image."
    (
        cd "$mnc_dir"
        ds_root=$(adjust_dataset_root "$ds_root")
        ./preprocess_kitti_mnc.sh "$ds_type" "$ds_root" "$ds_split" "$seq_id"
    ) || fail "Could not run MNC on sequence ID [$seq_id] under path: [$path]"
}

prepare_semantic () {
    prepare_semantic_mnc "$@"
}


# This is where the actual work starts.
ensure_sequence_ok "$DATASET_TYPE" "$DATASET_ROOT" "$DATASET_SPLIT" "$SEQUENCE_ID"
prepare_depth      "$DATASET_TYPE" "$DATASET_ROOT" "$DATASET_SPLIT" "$SEQUENCE_ID"
prepare_semantic   "$DATASET_TYPE" "$DATASET_ROOT" "$DATASET_SPLIT" "$SEQUENCE_ID"

echo
echo "Finished preprocessing [$DATASET_TYPE]-[$DATASET_SPLIT]-[$SEQUENCE_ID]"
echo "Dataset root:          [$DATASET_ROOT]"
echo
