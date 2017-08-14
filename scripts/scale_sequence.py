"""Processes the given KITTI sequence by reducing its resolution.

Computes depth maps (ELAS, dispnet) and semantic segmentations at the new low
resolution.
"""

import click
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


@click.group()
def cli():
    """Entry point for all subcommands."""
    pass

"""
Notes on the intrinsic matrix scaling => 
 * make this a parameter of DynSLAM
 * depending on whether it's set, it would read segmentation, depth, and color
 images from either the original folders, or the downscaled ones.
 * use this scale (0.75, 0.5, or 0.25, since time is limited), to 
"""

@cli.command()
@click.argument('dataset_root', type=click.File('r'))
@click.argument('sequence_type', type=str)
@click.argument('sequence_id', type=int)
@click.argument('scale', type=float)
def scale(sequence_folder, scale):
    # TODO implement
    # * collect list of left_rgb and right_rgb, scale them down, keep track of
    #   the names and locations of these downscaled frames, and then invoke
    #   ELAS (kitti2klg), DispNet (prolly directly from this script for
    #   simplicity), and MNC via something like the segment_tracking utility
    #   (which sets up env variables and stuff), but which just takes in
    #   arbitrary folders for input and output.
    # Note: it would take a while to run, mind you, so make sure the script
    # is reliable, and have it run over lunch/dinner or something. It is worth
    # investing time to make this script stable and clean.
    pass


if __name__ == "__main__":
    cli()
