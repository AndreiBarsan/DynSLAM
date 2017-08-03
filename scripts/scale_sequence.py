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


@cli.command()
@click.argument('sequence_folder', type=click.File('r'))
@click.argument('scale', type=float)
def scale(sequence_folder, scale):
    # TODO implement
    pass


if __name__ == "__main__":
    cli()
