# Instance Reconstruction Library (InstRecLib)

## Overview 

This is an extension of [InfiniTAM](https://github.com/victorprad/InfiniTAM) designed to work
directly with stereo input, segment the input based on semantics and motion, and then reconstruct
the static background of the scene, as well as all detected objects, separately.

The system is being developed as part of Andrei Bârsan's Master's Thesis at ETH Zurich. As such,
much of the functionality still relies on a bunch of custom external tools, such as [my fork
of MNC](https://github.com/AndreiBarsan/MNC), which can pre-compute segmentations for image
sequences (it also leverages an up-to-date version of Caffe, making it faster by supporting cuDNN
5).


## Building and Running

The system is built and set up just like
[the original InfiniTAM](https://github.com/victorprad/InfiniTAM), but currently relies on specific
pre-computed inputs for the depth and segmentation. They will be integrated into the main engine
over the coming months, but if you have any questions, please do not hesitate to open an issue
on [the project's GitHub page](https://github.com/AndreiBarsan/InfiniTAM).


## Miscellaneous

 * Note that this project's code uses 
   [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html#Declaration_Order),
   unlike InfiniTAM's core code, which uses other guidelines. 
 * See `TODO.md` for a big list of things to do.
 * If it's already past September 2017 and you're reading this, but can't find my write-up on this
   project, please open up an issue or e-mail me (the email is on my GitHub profile).
