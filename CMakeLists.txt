# Top-level CMake file for the DynSLAM project.
# TODO(andrei): Check out https://github.com/onqtam/ucm for tools to make cmake files nicer.

cmake_minimum_required(VERSION 3.1)
project(DynSLAM)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

macro(CANONIFY_BOOL var)
  if (${var})
    set(${var} TRUE)
  else()
    set(${var} FALSE)
  endif()
endmacro()

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

# We use Pangolin for the main GUI. Pangolin is a lightweight GUI library focused on visual
# applications with minimal support for multiple panels, widgets, and live plotting.
# Note that this part *must* be after the CUDA setup, lest you get errors about hidden symbols
# referenced by some DSO.
#add_subdirectory(src/Pangolin)   # TODO-LOW(andrei): Enabling this causes strange linker issues.
find_package(Pangolin REQUIRED)

# Use the Google flag library for sane handling of command-line flags.
add_subdirectory(src/gflags)
# Library for stereo visual odometry and sparse scene flow estimation.
add_subdirectory(src/libviso2)

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

include_directories(${Viso2_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# TODO(andrei): You may want to have these dependencies in src/DynSLAM/CMakeLists.txt.
add_subdirectory(src/InfiniTAM/InfiniTAM)
add_subdirectory(src/DynSLAM/InstRecLib)

# TODO(andrei): Make dynslam a library, and link the GUI with it.
set(DYNSLAM_SOURCES
    src/DynSLAM/ImageSourceEngine.cpp
    src/DynSLAM/Utils.cpp
    src/DynSLAM/DynSlam.cpp
    src/DynSLAM/InfiniTamDriver.cpp
    src/DynSLAM/Input.cpp
    src/DynSLAM/DepthProvider.h
    src/DynSLAM/PrecomputedDepthProvider.cpp
    src/DynSLAM/DSHandler3D.cpp

    # A tiny library for working with portable float map (PFM) files.
    src/pfmLib/ImageIOpfm.cpp)

set(DYNSLAM_GUI_SOURCES
    ${DYNSLAM_SOURCES}
    src/DynSLAM/DynSLAMGUI.cpp)

add_executable(DynSLAM ${DYNSLAM_SOURCES} ${EXTRA_EXECUTABLE_FLAGS} )

add_executable(DynSLAMGUI ${DYNSLAM_GUI_SOURCES} ${EXTRA_EXECUTABLE_FLAGS})
# TODO(andrei): Rename this to ITMEngine or something for clarity.
target_link_libraries(DynSLAMGUI Engine)
# TODO(andrei): Ditto. These are the InfiniTAM utilities!
target_link_libraries(DynSLAMGUI Utils)
target_link_libraries(DynSLAMGUI ITMLib)
target_link_libraries(DynSLAMGUI ORUtils)
target_link_libraries(DynSLAMGUI InstRecLib)
target_link_libraries(DynSLAMGUI ${Pangolin_LIBRARIES})
target_link_libraries(DynSLAMGUI ${OpenCV_LIBS})
target_link_libraries(DynSLAMGUI gflags)
#target_link_libraries(DynSLAMGUI ${Viso2_LIBS})
target_link_libraries(DynSLAMGUI viso2)
