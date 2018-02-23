FROM nvidia/cudagl:8.0-devel-ubuntu16.04
#FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04
LABEL maintainer andrei.ioan.barsan@gmail.com

# !!! IMPORTANT !!!
# This does NOT work yet. The dockerization effort is still a work in progress!
# Running OpenGL GUIs with CUDA in containers is non-trivial.

# Build and run this with 'nvidia-docker'. If you forget to do so, the build
# will NOT fail, but you will start getting strange issues when attempting
# to run Caffe or any of the tests.

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    vim \
    wget \
    libopencv-dev \
    libxmu-dev libxi-dev freeglut3 freeglut3-dev glew-utils libglew-dev libglew-dbg
#        && \
#    rm -rf /var/lib/apt/lists/*

# TODO why doesn't DynSLAM pick up the in-tree version of gflags?
RUN apt-get install -y --no-install-recommends \
    binutils-dev  \
    libgflags-dev \
    libpng++-dev \
    libpthread-stubs0-dev
#    gcc-5 g++-5

# TODO(andrei): Sort out the paths.
ENV DYNSLAM_ROOT=/opt/DynSLAM
WORKDIR $DYNSLAM_ROOT

ADD scripts        ./scripts
RUN scripts/install_cmake.sh sudo
RUN cmake --version

# TODO(andrei): It may be useful to make a tutorial about developing a CUDA image processing tool
# using a GUI with docker.

# TODO(andrei): Add src/DynSLAM separately, LAST, to minimize the number of required rebuilds...
ADD CMakeLists.txt ./
ADD src            ./src
#ADD data           ./data

RUN mkdir -p build/eigen    && cd build/eigen && cmake $DYNSLAM_ROOT/src/eigen    && make -j$(nproc)
RUN mkdir -p build/Pangolin && cd build/Pangolin && cmake $DYNSLAM_ROOT/src/Pangolin && make -j$(nproc)

# This has to be provided, since the capabilities test cannot run during Docker build.
ENV ITM_CUDA_COMPUTE_CAPABILITY=52

RUN apt-get install -y --no-install-recommends \
    binutils-dev  \
    mesa-utils

RUN mkdir -p build/DynSLAM && cd build/DynSLAM && \
    cmake $DYNSLAM_ROOT -DCUDA_COMPUTE_CAPABILITY=$ITM_CUDA_COMPUTE_CAPABILITY

# TODO(andrei): Preserve colors!
RUN cd build/DynSLAM && make -j$(nproc)

