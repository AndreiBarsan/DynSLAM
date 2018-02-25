# DynSLAM: Simultaneous Localization and Mapping in Dynamic Environments

This is a dense SLAM system written in C++. It builds on [InfiniTAM](https://github.com/victorprad/InfiniTAM), adding support
for stereo input, outdoor operation, voxel garbage collection,
and separate dynamic object (e.g., car) reconstruction.

Developed as part of my Master's Thesis, in the [Computer
Vision and Geometry Group](https://cvg.ethz.ch) of [ETH
Zurich](https://ethz.ch). Accepted to ICRA 2018 accompanying
the paper "Robust Dense Mapping for Large-Scale Dynamic 
Environments" by Andrei Bârsan, Peidong Liu, Marc Pollefeys, and Andreas Geiger.

The source code is [hosted on GitHub](https://github.com/AndreiBarsan/DynSLAM).

## Preview

The following screenshot shows an early preview of DynSLAM in action. It
takes in stereo input, computes the depth map, using either ELAS or
dispnet, segments the input RGB using Multi-task Network Cascades to
detect object instances, and then separately reconstructs the static
background and individual object instances.

The top pane shows the dense reconstruction of the background. The
following panes show, in top-down, left-right order: the left RGB frame,
the computed depth map, the output of the instance-aware semantic
segmentation algorithm, the input RGB to the instance reconstructor,
memory usage statistics, and a novel view of the reconstructed object
instance.

The colors in the 3D reconstructions correspond to the voxel weights:
red-tinted areas are low-weight ones, whereas blue ones are high-weight
ones. Areas which remain low-weight even several frames after first
being observed are very likely to be noisy, while blue ones are ones
where the system is confident in its reconstruction.

![DynSLAM GUI screenshot](data/screenshots/dynslam-preview.png)

## Related Repositories

 * [My InfiniTAM fork](https://github.com/AndreiBarsan/InfiniTAM), which
   is used by this system for the actual 3D reconstruction (via
   volumetric fusion, using voxel hashing for map storage). My fork
   contains a series of small tweaks designe to make InfiniTAM a little
   easier to use as a component of a larger system.
 * [My fork of the official implemntation of Multi-task Network Cascades](https://github.com/AndreiBarsan/MNC)
    for image semantic segmentation. We need this for identifying where
    the cars are in the input videos. Using semantics enables us to
    detect both moving and static cars.
 * [My fork of the modified Caffe used by MNC](https://github.com/AndreiBarsan/caffe-mnc). Since MNC's architecture requires
 some tweaks to Caffe's internals, its authors forked Caffe and modified
 it to their needs. I forked their fork and made it work with my tools,
 while also making it faster by merging it with the Caffe master, which
 enabled cuDNN 5 support, among many other things.
  * [My mirror of libelas](https://github.com/AndreiBarsan/libelas-tooling)
  which I use for pre-computing the depth maps. I'm working on getting
  the depth computation to happen on the fly, and investigating other
  methods for estimating depth from stereo.

## Regenerating Plots

The plots in the corresponding ICRA paper can all be regenerated from the raw
data included in this repository as follows:

  1. Unzip `./raw-data-archives/raw-logz.7z` to `./csv`.
  1. Install the data analysis dependencies (e.g., in a Python virtual
     environment or using Anaconda). Installing the pacakges using the Anaconda
     option can be done as:
     ```bash
     conda install --yes jupyter pandas numpy scipy scikit-learn matplotlib seaborn
     ```
  1. Start Jupyter:
     ```bash
     cd notebooks && jupyter notebook
     ```
  1. Regenerate Figure 6 using `./notebooks/StaticAndDynamicDepthAnalysis.ipynb`
  1. Regenerate Figure 7 using `./notebooks/Voxel GC Stats.ipynb`
  1. The other notebooks can be used to generate the various figures from [the
     supplementary material](http://andreibarsan.github.io/dynslam).


## Building and Running DynSLAM

If you want to check out the system very quickly, you're in luck!
There's a pre-preprocessed sequence you can download to see how it works (see 
the "Demo Sequence" section).

If you want to preprocess your own sequences, see the "Preprocessing" section.

### Building 

This project is built using CMake, and it depends on several submodules. 
As such, make sure you don't forget the `--recursive` flag when cloning the 
repository. If you did
forget it, just run `git submodule update --init --recursive`.

 1. Clone the repository if you haven't already:
    ```bash
    git clone --recursive https://github.com/AndreiBarsan/DynSLAM
    ```
 1. Install OpenCV 2.4.9 and CUDA 8.
 1. [Install docker and nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
    They are a requirement for preprocessing the data so that it can be consumed
    by DynSLAM.
 1. Install the prerequisites (Ubuntu example):
    ```bash
    sudo apt-get install libxmu-dev libxi-dev freeglut3 freeglut3-dev glew-utils libglew-dev libglew-dbg libpthread-stubs0-dev binutils-dev libgflags-dev libpng++-dev libeigen3-dev
    ```
 1. Build Pangolin to make sure it gets put into the CMake registry:
    ```bash
    cd src/Pangolin && mkdir build/ && cd $_ && cmake ../ && make -j$(nproc)
    ```
 1. Build the project in the standard CMake fashion:
    ```bash
    mkdir build && cd build && cmake .. && make -j$(nproc)
    ```
    
### Building DynSLAM Inside Docker

While the preprocessing makes heavy use of `nvidia-docker` in order to simplify
the process, and does so very effectively, running the main DynSLAM program 
inside Docker is still not supported.

The `Dockerfile` in the root of this project *can* be used to build DynSLAM 
inside a Docker container, but, due to its OpenGL GUI, it cannot run inside it
(as of February 2018).

Solutions to this problem include using one of the newly released CUDA+OpenGL 
Docker images from NVIDIA as a base image, or fully supporting CLI-only 
operation. Both of these tasks remain part of future work.


### Demo Sequence
 1. After building the project, try processing the demo sequence: 
    [here is a short sample from KITTI Odometry Sequence 06](http://www.cs.toronto.edu/~iab/dynslam/mini-kitti-odometry-seq-06-for-dynslam.7z).
      1. Extract that to a directory, and run DynSLAM on it (the mkdir circumvents a silly bug):
        ```bash
        mkdir -p csv/ && build/DynSLAM --use_dispnet --dataset_root=path/to/extracted/archive --dataset_type=kitti-odometry
        ```

### KITTI Tracking and Odometry Sequences
 1. The system can run on any KITTI Odometry and Tracking sequence. 
    KITTI Raw sequences should also work, but have not been 
    tested since the evaluation is trickier, as their LIDAR data is not cleaned
    up to account for the rolling shutter effect. In order to run the system on
    these sequences, the instance-aware semantic segmentations and dense depth
    maps must be preprocessed, since DynSLAM does not yet support computing them
    on the fly. 
    
    These instructions are for the KITTI Tracking dataset, which is
    the only one currently supported using helper scripts, but I plan on adding
    support for easy KITTI Odometry preprocessing, since the only difference
    between the two datasets is the path structure.
    1. Use the following download script to grab the KITTI Tracking dataset. Bear in mind
       that it is a very large dataset which takes up a little over 100Gb of
       disk space. Sadly, the download is structured such that downloading 
       individual sequences is not possible.
       ```bash
       scripts/download_kitti_tracking.py [target-dir]
       ```
       By default, this script downloads the data to the `data/kitti/tracking`
       folder of the DynSLAM project.
    1. (Alternative) You can also manually grab the KITTI Tracking dataset 
    [from the official website](www.cvlibs.net/datasets/kitti/eval_odometry.php).
    Make sure you download everything and extract it all in the same directory 
    (see the demo sequence archive for the canonical directory structure, or 
    `Input.h` to see how DynSLAM loads it).
    1. Preprocess the data using the preprocessing script:
        ```bash
        scripts/preprocess-sequence.sh kitti-tracking <dataset-root> <training/testing> <number>
        ```
        For example,
        ```bash
        scripts/preprocess-sequence.sh kitti-tracking data/kitti/tracking training 6
        ```
        prepares the 6th KITTI Tracking training sequence for DynSLAM.
    1. Run the pipeline on the KITTI sequence you downloaded.
       ```bash
       ./DynSLAM --use_dispnet --enable-evaluation=false --dataset_root=<dataset-root> --dataset_type=kitti-tracking --kitti_tracking_sequence_id=<number>
       ```
 
 You can also use `DynSLAM --help` to view info on additional commandline arguments. (There are a lot of them!)

## Remarks

  * The code follows
    [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html)
    with the following modifications:

    * The column limit is 100 instead of 80, because of the bias towards
      longer type names in the code base.
    * Exceptions are allowed, but must be used judiciously (i.e., for
      serious errors and exceptional situations).
