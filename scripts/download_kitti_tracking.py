#!/usr/bin/env python
""""Downloads and unzips the KITTI tracking data.

Warning: This can take a while, and use up >100Gb of disk space."""

from __future__ import print_function

import argparse
import os
import sys

from subprocess import call


# Odometry files
#  - data_odometry_gray.zip
#  - data_odometry_color.zip
#  - data_odometry_velodyne.zip
#  - data_odometry_calib.zip

URL_BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
tracking_dir_names = ['image_02', 'image_03', 'velodyne', 'calib']
tracking_dir_zip_tags = ['image_2', 'image_3', 'velodyne', 'calib']
odo_dir_names = ['gray', 'color', 'velodyne', 'calib']

# TODO(andrei): Maybe remove
LEFT_COLOR_FNAME="data_tracking_image_2.zip"
RIGHT_COLOR_FNAME="data_tracking_image_3.zip"
VELODYNE_FNAME="data_tracking_velodyne.zip"
CALIB_FNAME="data_tracking_calib.zip"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_root', type=str, default=os.path.join('data', 'kitti'))
    # parser.add_argument('--root', type=str, default=None, help='data folder')

    return parser.parse_args(sys.argv[1:])


def main():
    args = parse_args()
    kitti_dir = args.kitti_root

    # Perform a few sanity checks to make sure we're operating in the right dir
    # when left with the default args.
    if not os.path.isabs(kitti_dir):
        if not os.path.isdir('src'):
            os.chdir('..')

            if not os.path.isdir('src'):
                print("Please make sure to run this tool from the DynSLAM "
                      "project root when using relative paths.")
                return 1

    tracking_dir = os.path.join(kitti_dir, 'tracking')
    os.makedirs(tracking_dir, exist_ok=True)
    os.chdir(tracking_dir)

    tracking_zip_names = ["data_tracking_" + name + ".zip" for name in tracking_dir_zip_tags]

    for dir_name, zip_name in zip(tracking_dir_names, tracking_zip_names):
        canary_dir = os.path.join('training', dir_name)
        if os.path.isdir(canary_dir):
            print("Directory {} canary dir seems to exist, so I will assume the data is there.".format(canary_dir))
        else:
            if os.path.exists(zip_name):
                print("File {} exists. Not re-downloading.".format(zip_name))
            else:
                url = URL_BASE + zip_name
                print("Downloading file {} to folder {}.".format(zip_name, kitti_dir))
                call(['wget', url])

            call(['unzip', zip_name])

    return 0


if __name__ == '__main__':
    sys.exit(main())
