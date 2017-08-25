#!/usr/bin/env python3
"""Processes the given KITTI sequence by reducing its resolution.

Computes depth maps (ELAS, dispnet) and semantic segmentations at the new low
resolution.

Note: this script is quite inconsistent and hacky, as it was written under very
tight time constraints. :(

Note: you shouldn't care that the ELAS tool produces depth maps, and that the
DispNet one produces disparity maps. DynSLAM takes care of this stuff either
way.
"""

import os
import subprocess
import sys

import click
import scipy
from scipy import misc


KITTI_ODOMETRY = 'kitti-odometry'
KITTI_TRACKING = 'kitti-tracking'

# While spooky-looking, all these tools can be set up from their appropriate
# repos on my GitHub (github.com/AndreiBarsan). See the rest of this script for
# more details!
ELAS_BATCH_BINARY = '/home/barsana/work/libelas/cmake-build-debug/kitti2klg'
DISPNET_DIR = '/home/barsana/work/dispnet-flownet-docker/'
DISPNET_PROCESS_KITTI_ODOMETRY = 'process-kitti.sh'
DISPNET_PROCESS_KITTI_TRACKING = 'process-kitti-tracking.sh'
MNC_DIR = '/home/barsana/work/MNC/'
MNC_SCRIPT = 'run-seg.sh'


"""
Notes on the intrinsic matrix scaling => 
 * make this a parameter of DynSLAM
 * depending on whether it's set, it would read segmentation, depth, and color
 images from either the original folders, or the downscaled ones.
 * use this scale (0.75, 0.5, or 0.25, since time is limited), to adjust the
   camera intrinsics (location of princ.point, focal length)
"""

# Path patterns for files from the KITTI odometry dataset.
odometry_pat = {
    'rgb_l': "{root}/sequences/{sequence_id:02d}/image_2/",
    'rgb_r': "{root}/sequences/{sequence_id:02d}/image_3/",
    'rgb_l_scaled': "{root}/sequences/{sequence_id:02d}/image_2_{scale:.2f}/",
    'rgb_r_scaled': "{root}/sequences/{sequence_id:02d}/image_3_{scale:.2f}/",
    'rgb_file': '{img_id:06d}.png',

    'depth_elas_scaled': "{root}/sequences/{sequence_id:02d}/precomputed-depth-elas-{scale:.2f}/",
    # 'depth_dispnet_scaled': "{root}/sequences/{sequence_id:02d}/precomputed-depth-dispnet-{scale:.2f}/",
    'seg_scaled': "{root}/sequences/{sequence_id:02d}/seg_image_2-{scale:.2f}/mnc",
}

# Path patterns for files from the KITTI tracking dataset.
tracking_pat = {
    'rgb_l': "{root}/training/image_02/{sequence_id:04d}/{img_id:06d}.png",
    'rgb_r': "{root}/training/image_03/{sequence_id:04d}/{img_id:06d}.png",
    # etc...
}

seq_to_pat = {
    KITTI_ODOMETRY: odometry_pat,
    KITTI_TRACKING: tracking_pat
}


def mkdirp(path):
    """Emulates `mkdir -p` functionality."""
    try:
        os.makedirs(path)
    except FileExistsError:
        # The folder already exists. We're good.
        pass


def resize(in_path_pattern, out_path_pattern, file_pattern, root, sequence_id,
           scale, force):
    in_dir = in_path_pattern.format(root=root, sequence_id=sequence_id)
    out_dir = out_path_pattern.format(root=root, sequence_id=sequence_id,
                                       scale=scale)

    if os.path.isdir(out_dir) and not force:
        print("The resize output folder [{}] already exists. "
              "Not performing the resize.".format(out_dir))
        return

    mkdirp(out_dir)
    for img_fname in sorted(os.listdir(in_dir)):
        in_fpath = os.path.join(in_dir, img_fname)
        out_fpath = os.path.join(out_dir, img_fname)

        print("Input: {}".format(in_fpath))
        print("Output: {}".format(out_fpath))

        img = scipy.misc.imread(in_fpath)
        resized = scipy.misc.imresize(img, scale)
        scipy.misc.imsave(out_fpath, resized)


def compute_elas_depth_odo(patterns, root, sequence_id, scale, force):
    """Processes left and right RGB frames to produce depth maps using libelas.

    Designed to leverage the 'kitti2klg' utility available here:
    https://github.com/AndreiBarsan/libelas-tooling
    """
    odo_root = os.path.join(root, 'sequences', '{:02d}'.format(sequence_id))
    out_dir = patterns['depth_elas_scaled'].format(root=root, sequence_id=sequence_id, scale=scale)

    if os.path.isdir(out_dir) and not force:
        print("The ELAS output folder [{}] already exists. "
              "Not performing the computation.".format(out_dir))
        return

    mkdirp(out_dir)
    subprocess.call([
        ELAS_BATCH_BINARY,
        '--baseline_meters=0.537150654273',
        '--kitti_root={}'.format(odo_root),
        '--scale={}'.format(scale),
        '--infinitam',
        '--output={}'.format(out_dir),
        '--use_color',
        '--calib_file=calib.txt'
    ])


def compute_elas_depth_tracking(patterns, root, sequence_id, scale, force):
    raise ValueError("TODO(andrei): Implement this.")


def compute_dispnet_disparity_odo(patterns, root, sequence_id, scale, force):
    """Processes left and right RGB frames to produce disparity maps using DispNet.

    See: https://github.com/AndreiBarsan/dispnet-flownet-docker
    """
    sequence_root_dir = os.path.join(root, 'sequences', '{:02d}'.format(sequence_id))
    dispnet_out_dir = "precomputed-depth-dispnet-{:.2f}".format(scale)
    dispnet_out_dir_full = os.path.join(sequence_root_dir, dispnet_out_dir)

    if os.path.isdir(dispnet_out_dir_full) and not force:
        print("The DispNet output folder [{}] already exists. "
              "Not performing the computation.".format(dispnet_out_dir))
        return

    left_subdir = "image_2_{:.2f}".format(scale)
    right_subdir = "image_3_{:.2f}".format(scale)

    mkdirp(dispnet_out_dir_full)
    return subprocess.call([
            os.path.join(DISPNET_DIR, DISPNET_PROCESS_KITTI_ODOMETRY),
            sequence_root_dir,
            left_subdir,
            right_subdir,
            dispnet_out_dir
        ],
        cwd=DISPNET_DIR)


def compute_dispnet_disparity_tracking(patterns, root, sequence_id, scale, force):
    raise ValueError("TODO(andrei): Implement this.")


def compute_mnc_segmentation(patterns, root, sequence_id, scale, force):
    mnc_in_dir = patterns['rgb_l_scaled'].format(root=root,
                                                 sequence_id=sequence_id,
                                                 scale=scale)
    mnc_out_dir = patterns['seg_scaled'].format(root=root,
                                                sequence_id=sequence_id,
                                                scale=scale)

    if os.path.isdir(mnc_out_dir) and not force:
        print("The MNC output folder [{}] already exists. Not performing the "
              "computation.".format(mnc_out_dir))
        return

    mkdirp(mnc_out_dir)
    return subprocess.call([
        os.path.join(MNC_DIR, MNC_SCRIPT),
        '--input', mnc_in_dir,
        '--output', mnc_out_dir
    ], cwd=MNC_DIR)


@click.command()
@click.argument('sequence_type', type=str)
@click.argument('dataset_root', type=click.Path(exists=True, dir_okay=True,
                                                file_okay=False,
                                                resolve_path=True))
@click.argument('sequence_id', type=int)
@click.argument('scale', type=float)
@click.option('--force/--no-force', default=False,
              help="Whether to force resizing/computations even when results "
                   "are already present.")
@click.option('--skip-resize/--no-skip-resize', default=False)
@click.option('--skip-elas/--no-skip-elas', default=False)
@click.option('--skip-dispnet/--no-skip-dispnet', default=False)
@click.option('--skip-semantics/--no-skip-semantics', default=False)
def scale(dataset_root, sequence_type, sequence_id, scale, force, skip_resize,
          skip_elas, skip_dispnet, skip_semantics):
    """Scales down a KITTI sequence (tracking or odometry).

    Used to evaluate the behavior of DynSLAM on lower-resolution input.
    """

    if sequence_type not in seq_to_pat:
        raise ValueError("Unknown sequence type [{}]. Supported are [{}].".format(
            sequence_type, seq_to_pat.values()
        ))

    if scale >= 1.0:
        raise ValueError("The scale is not meant to be >= 1.0. Blowing up "
                         "images is not supported.")

    patterns = seq_to_pat[sequence_type]

    if not skip_resize:
        resize(patterns['rgb_l'], patterns['rgb_l_scaled'], patterns['rgb_file'],
               dataset_root, sequence_id, scale, force)
        resize(patterns['rgb_r'], patterns['rgb_r_scaled'], patterns['rgb_file'],
               dataset_root, sequence_id, scale, force)
    else:
        print("Skipping image resizing.")

    if not skip_elas:
        if sequence_type == KITTI_ODOMETRY:
            compute_elas_depth_odo(patterns, dataset_root, sequence_id, scale,
                                   force)
        elif sequence_type == KITTI_TRACKING:
            compute_elas_depth_tracking(patterns, dataset_root, sequence_id,
                                        scale, force)
        else:
            raise ValueError("Sanity check failed: unknown sequence type "
                             "{}".format(sequence_type))
    else:
        print("Skipping computation of ELAS depth maps.")

    if not skip_dispnet:
        if sequence_type == KITTI_ODOMETRY:
            compute_dispnet_disparity_odo(patterns, dataset_root, sequence_id,
                                          scale, force)
        elif sequence_type == KITTI_TRACKING:
            compute_dispnet_disparity_tracking(patterns, dataset_root,
                                               sequence_id, scale, force)
        else:
            raise ValueError("Sanity check failed: unknown sequence type "
                             "{}".format(sequence_type))
    else:
        print("Skipping computation of DispNet disparity maps.")

    if not skip_semantics:
        compute_mnc_segmentation(patterns, dataset_root, sequence_id, scale, force)
    else:
        print("Skipping computation of instance-aware semantic segmentations.")


if __name__ == "__main__":
    scale()
