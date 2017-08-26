#!/usr/bin/env bash
# A script for preparing a KITTI odometry sequence for DynSLAM.
#
# Unfinished. Only sets up ground truth VO at the moment, which is not a
# key part, since libviso2 works fine.
# TODO(andrei): Finish implementing in a fully generic way.

set -euo pipefail
IFS=$'\n\t'

usage() {
  echo >&2 "Usage: $0 <kitti-odometry-dataset-root>"
}

if [[ "$#" -ne 1 ]]; then
  usage;
  exit 1;
fi

DATASET_ROOT="$1"
SEQS="${DATASET_ROOT}/sequences"
POSES="${DATASET_ROOT}/poses"

echo "Dataset root: $DATASET_ROOT"

if ! [[ -d "${SEQS}" ]]; then
  echo >&2 "No sequences directory found. Looking for: ${SEQS}"
  exit 2;
fi

if ! [[ -d "${POSES}" ]]; then
  echo >&2 "No ground truth poses directory found. Looking for: ${POSES}"
  exit 3;
fi

for SEQ_DIRNAME in $(ls "${SEQS}"); do
  GT_POSES="${POSES}/${SEQ_DIRNAME}.txt"
  SEQ_DIR="${SEQS}/${SEQ_DIRNAME}"
  echo "Processing sequence ${SEQ_DIRNAME}."
  if [[ -f "${GT_POSES}" ]]; then
    ln -sf "${GT_POSES}" "${SEQ_DIR}/ground-truth-poses.txt"
  else
    echo "Warning: No ground truth poses available for sequence [${SEQ_DIR}]."
  fi
done

echo "Done."

