# DynSLAM: Simultaneous Localization and Mapping in Dynamic Environments

This is a dense SLAM system written in C++. It builds on [InfiniTAM](https://github.com/victorprad/InfiniTAM), adding support
for stereo input and separate dynamic object (e.g., car) reconstruction.

Currently under development as my Master's Thesis, as part of the [Computer
Vision and Geometry Group](https://cvg.ethz.ch) of [ETH
Zurich](https://ethz.ch).

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

## Getting Started

Coming soon! Right now the system is a bit tangled up, so it can't be run out
of the box without jumping through a lot of hoops. This will change over the 
course of the next few months.

### (Currently not Working) Steps

Important: if you're interested in this project and it's after September 1st
2017, please email me! My email is on my GitHub profile page. I will update the
instructions accordingly. Reproducibility is VERY important to me.

Note that the system is under *heavy* development at the moment, so that these
instructions could quickly go out of date. Generally speaking, this project is
built using CMake, and it depends on several submodules. As such, make sure you
don't forget the `--recursive` flag when cloning the repository. If you did
forget it, just run `git submodule update --init --recursive`.

 1. Clone the repository if you haven't already:
 ```bash
 git clone --recursive https://github.com/AndreiBarsan/DynSLAM
 ```
 2. Install OpenCV 2.4.9 and CUDA (no special version requirements at the moment).
 
 3. Install the remaining prerequisites (Ubuntu example):
 ```bash
 sudo apt-get install libxmu-dev libxi-dev freeglut3 freeglut3-dev glew-utils libglew-dev libglew-dbg
 ```
 
 4. Build the project in the standard CMake fashion:
 ```bash
 mkdir build && cd build && cmake .. && make -j
 ```
 5. Grab any raw KITTI data sequence [from the official website](http://www.cvlibs.net/datasets/kitti/raw_data.php). Make sure it's a synced+rectified
 sequence.
 
 6. Use the [MNC pre-trained neural network](http://github.com/AndreiBarsan/MNC)
    to process the KITTI sequence. In the future, this will be integrated into
    the main pipeline but right now Caffe is a bit capricious.
    
 7. (TODO; right now many things are hard-coded, sorry :( ) Run the pipeline on
 the KITTI sequence you downloaded.
 ```bash
 ./DynSLAM path/to/kitti/sequence
 ```

## Remarks

  * The code follows
    [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html)
    with the following modifications:

    * The column limit is 100 instead of 80, because of the bias towards
      longer type names in the code base.
    * Exceptions are allowed, but must be used judiciously (i.e., for
      serious errors and exceptional situations).
