# TODO list for the Instance Reconstruction Library (InstRecLib)

## Overview
 * The project has started as an extension of InfiniTAM, but the goal is to make it generic enough
 so it can operate with any underlying SLAM framework, such as ElasticFusion, Kintinuous, gtsam, etc.
 * In order to achieve this, the code must be written in such a way that it only depends on
 InfiniTAM via adapter classes, which can easily be swapped out to allow working with a different
 base system. These dependencies currently include InfiniTAM-specific image manipulation routines,
 and the ITMView data wrapper.
 
## Tasks

 - [ ] Decide on InstRecLib-specific internal image/data representation. Should we really depend on OpenCV for this? OpenCV is a nasty dependency to have just for a little IO. But, then again, if we end up having to integrate tightly with Caffe, it requires OpenCV anyway...
 - [ ] Ask Peidong if we can simply depend on certain InfiniTAM utilities, if we factor them out
       into a separate library. For example, the image classes, MemoryBlock, Matrix{3,4} etc., are
       very well-written and flexible. I could simply rewrite them in my own library, but they would
       end up being VERY similar.
 - [ ] View (ITMView) adapter class.
 - [ ] Image adapter class.
 - [ ] Custom MemoryBlock class?
 - [ ] Custom Matrix classes?
 - [ ] Custom Vector classes?
 - [ ] Ensure headers are lean.
 - [ ] clang-format or equivalent IDE-agnostic code formatter
 - [ ] Ensure code formatter also enforces naming conventions.
 - [ ] Maybe rename namespaces to be all-lowercase.
 - [ ] Implement CUDA version of instance masking.
 