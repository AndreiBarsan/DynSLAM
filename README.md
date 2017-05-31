# DynSLAM: Simultaneous Localization and Mapping in Dynamic Environments

This is a dense SLAM system written in C++. It builds on [InfiniTAM](https://github.com/victorprad/InfiniTAM), adding support
for stereo input and separate dynamic object (e.g., car) reconstruction.

Currently under development as my Master's Thesis, as part of the [Computer
Vision and Geometry Group](https://cvg.ethz.ch) of [ETH
Zurich](https://ethz.ch).

## Getting Started

Coming soon! Right now the system is a bit tangled up, so it can't be run out
of the box without jumping through a lot of hoops. This will change over the 
course of the next few months.

### (Currently not Working) Steps

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
