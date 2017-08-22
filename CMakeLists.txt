# Top-level CMake file for the DynSLAM project.
# TODO(andrei): Check out https://github.com/onqtam/ucm for tools to make cmake files nicer.

cmake_minimum_required(VERSION 3.1)
project(DynSLAM)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

macro(CANONIFY_BOOL var)
  if (${var})
    set(${var} TRUE)
  else()
    set(${var} FALSE)
  endif()
endmacro()

# If enabled, enables sensible stack traces on Linux, complete with corresponding source
# code, where available. CUDA errors also produce complete stack traces when this is on.
# If disabled, the error messages degrade gracefully to file/line information.
OPTION(WITH_BACKWARDS_CPP "Build with backwards-cpp stack trace dumping library? (Linux-only)" TRUE)
message(STATUS "backwards-cpp-enhanced stack traces? " ${WITH_BACKWARDS_CPP})

# We use Eigen for generic matrix support and linear algebra routines.
find_package(Eigen3 REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

find_package(CUDA QUIET)
CANONIFY_BOOL(CUDA_FOUND)
message(STATUS "CUDA found: " ${CUDA_FOUND})
# TODO(andrei): Is this optional any more?
OPTION(WITH_CUDA "Build with CUDA support?" ${CUDA_FOUND})

IF(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ELSE(MSVC_IDE)
  set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
    # TODO(andrei): Do we still need the '-g' and '-O' flags with -DCMAKE_BUILD_TYPE(=debug) flag?
#  set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native -g ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF(MSVC_IDE)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
endif(APPLE)

IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -Wno-deprecated-declarations -Wno-unused-function")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++")
ENDIF()

# Set some additional common compilation flags, like C++11 feature support.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# TODO(andrei): Maybe just remove it from the ITM tree and put it in src/?
if(WITH_BACKWARDS_CPP)
  add_definitions(-DWITH_BACKWARDS_CPP)
  add_subdirectory(src/InfiniTAM/InfiniTAM/ORUtils/third_party/backward-cpp)
endif()

# We use Pangolin for the main GUI. Pangolin is a lightweight GUI library focused on visual
# applications with minimal support for multiple panels, widgets, and live plotting.
# Note that this part *must* be after the CUDA setup, lest you get errors about hidden symbols
# referenced by some DSO.
#add_subdirectory(src/Pangolin)   # TODO-LOW(andrei): Enabling this causes strange linker issues.
set(BUILD_PANGOLIN_EIGEN ON CACHE BOOL "")
find_package(Pangolin REQUIRED)

# Use the Google flag library for sane handling of command-line flags.
add_subdirectory(src/gflags)
# Library for stereo visual odometry and sparse scene flow estimation.
add_subdirectory(src/libviso2)
# Work with Lie Groups using Eigen3
add_subdirectory(src/Sophus)
find_package(Sophus REQUIRED)

IF(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
ELSE()
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
ENDIF()

# We rely on OpenCV for image IO and manipulation. Moreover, our (future) Caffe
# dependencies will require it anyway...
find_package(OpenCV REQUIRED)

# TODO(andrei): See pangolin's 'external' folder for neat ways of bringing in dependencies
# from GitHub and building them if necessary.

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Viso2_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})

# TODO(andrei): You may want to have these dependencies in src/DynSLAM/CMakeLists.txt.
add_subdirectory(src/InfiniTAM/InfiniTAM)
add_subdirectory(src/DynSLAM/InstRecLib)

# Typically you don't have to add headers as sources, since they get included and compiled
# anyway, but CLion's static analysis works better on them if they're listed as sources.
set(DYNSLAM_SOURCES
    # A tiny library for working with portable float map (PFM) files.
    src/pfmLib/ImageIOpfm.cpp

    # Experimental
    src/DynSLAM/Direct/pinholeCameraModel.cpp
    src/DynSLAM/Direct/helperFunctions.cpp
    src/DynSLAM/Direct/feature/feature_depthMap.cpp
    src/DynSLAM/Direct/frame/frame.cpp
    src/DynSLAM/Direct/robustLoss/pseudoHuberLoss.cpp
    src/DynSLAM/Direct/robustLoss/tDistributionLoss.cpp
    src/DynSLAM/Direct/transformation/transformation.cpp
    # TODO(andrei): Support CUDA directly in DynSLAM project, and build the CUDA frame as well.
    src/DynSLAM/Direct/frame/device/cpu/frame_cpu.cpp
    src/DynSLAM/Direct/image_alignment/dirImgAlignBase.cpp
    src/DynSLAM/Direct/image_alignment/device/cpu/dirImgAlignCPU.cpp
    # End experimental parts

    src/DynSLAM/Evaluation/EvaluationCallback.cpp
    src/DynSLAM/Evaluation/EvaluationCallback.h
    src/DynSLAM/DepthProvider.h
    src/DynSLAM/DSHandler3D.cpp
    src/DynSLAM/DynSlam.cpp
    src/DynSLAM/Evaluation/CsvWriter.cpp
    src/DynSLAM/Evaluation/CsvWriter.h
    src/DynSLAM/Evaluation/ErrorVisualizationCallback.cpp
    src/DynSLAM/Evaluation/ErrorVisualizationCallback.h
    src/DynSLAM/Evaluation/Evaluation.cpp
    src/DynSLAM/Evaluation/ILidarEvalCallback.h
    src/DynSLAM/Evaluation/Tracklets.cpp
    src/DynSLAM/Evaluation/Tracklets.h
    src/DynSLAM/Evaluation/VelodyneIO.cpp
    src/DynSLAM/InfiniTamDriver.cpp
    src/DynSLAM/Input.cpp
    src/DynSLAM/PrecomputedDepthProvider.cpp
    src/DynSLAM/PrecomputedDepthProvider.h
    src/DynSLAM/Utils.cpp src/DynSLAM/Evaluation/SegmentedEvaluationCallback.cpp src/DynSLAM/Evaluation/SegmentedEvaluationCallback.h src/DynSLAM/Evaluation/Records.h src/DynSLAM/Evaluation/SegmentedCallback.cpp src/DynSLAM/Evaluation/SegmentedCallback.h src/DynSLAM/Evaluation/SegmentedVisualizationCallback.cpp src/DynSLAM/Evaluation/SegmentedVisualizationCallback.h)

set(DYNSLAM_GUI_SOURCES
    ${DYNSLAM_SOURCES}
    src/DynSLAM/DynSLAMGUI.cpp)

add_library(DynSLAM ${DYNSLAM_SOURCES} ${EXTRA_EXECUTABLE_FLAGS})
# TODO(andrei): Rename this to ITMEngine or something for clarity.
target_link_libraries(DynSLAM Engine)
# TODO(andrei): Ditto. These are the InfiniTAM utilities!
target_link_libraries(DynSLAM Utils)
target_link_libraries(DynSLAM ITMLib)
target_link_libraries(DynSLAM ORUtils)
target_link_libraries(DynSLAM InstRecLib)
target_link_libraries(DynSLAM ${OpenCV_LIBS})
target_link_libraries(DynSLAM gflags)
target_link_libraries(DynSLAM ${Viso2_LIBS})

add_executable(DynSLAMGUI ${DYNSLAM_GUI_SOURCES} ${EXTRA_EXECUTABLE_FLAGS})
target_link_libraries(DynSLAMGUI ${Pangolin_LIBRARIES})
target_link_libraries(DynSLAMGUI Engine)
target_link_libraries(DynSLAMGUI Utils)
target_link_libraries(DynSLAMGUI ITMLib)
target_link_libraries(DynSLAMGUI ORUtils)
target_link_libraries(DynSLAMGUI InstRecLib)
target_link_libraries(DynSLAMGUI ${OpenCV_LIBS})
target_link_libraries(DynSLAMGUI gflags)
target_link_libraries(DynSLAMGUI ${Viso2_LIBS})

if(WITH_BACKWARDS_CPP)
  # Link agains libbfd to ensure backward-cpp can extract additional information from the binary,
  # such as source code mappings. The '-lbfd' dependency is optional, and if it is disabled, the
  # stack traces will still work, but won't show unmangled symbol names or source code snippets.
  # You may need to set BACKWARD_USE_BFD to 0 in its `hpp` and `cpp` files to avoid linker errors.
  target_link_libraries(DynSLAMGUI -lbfd)
  target_link_libraries(DynSLAMGUI backward)
endif()

