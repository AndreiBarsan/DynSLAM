cmake_minimum_required(VERSION 3.1)
project(DynSLAM)

# TODO(andrei): Print TODOs report tool.

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

# We rely on OpenCV for image IO and manipulation. Moreover, our (future) Caffe
# dependencies will require it anyway...
find_package(OpenCV REQUIRED)

# TODO(andrei): Is this optional any more?
OPTION(WITH_CUDA "Build with CUDA support?" ${CUDA_FOUND})

OPTION(WITH_BACKWARDS_CPP "Build with backwards-cpp stack trace dumping library? (Linux-only)" TRUE)
message(STATUS "backwards-cpp-enhanced stack traces? " ${WITH_BACKWARDS_CPP})

IF(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ELSE(MSVC_IDE)
  set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
  # TODO(andrei): Proper debug build config.
#  set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fPIC -g ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
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

#if(WITH_BACKWARDS_CPP)
  #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lbfd")
  #add_definitions(-DWITH_BACKWARDS_CPP)
#ENDIF()

# TODO(andrei): You may want to have this dependency in src/DynSLAM/CMakeLists.txt.
add_subdirectory(src/InfiniTAM/InfiniTAM)
# TODO(andrei): Enable this and build.
# TODO(andrei): Then, slowly move its functionality in this parent project, and
# keep InfiniTAM pure.
#add_subdirectory(src/InfiniTAM/InfiniTAM/InstRecLib)
#add_subdirectory(src/DynSLAM)

IF(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
ELSE()
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
ENDIF()

include_directories(${OpenCV_INCLUDE_DIRS})

if(WITH_BACKWARDS_CPP)
  set(EXTRA_EXECUTABLE_FLAGS ${BACKWARD_ENABLE})
  # We need libbfd for allowing backwards-cpp to read lots of useful
  # debug data from our binaries.
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lbfd")
  add_definitions(-DWITH_BACKWARDS_CPP)
ELSE()
  set(EXTRA_EXECUTABLE_FLAGS "")
ENDIF()

set(DYNSLAM_SOURCES
    src/DynSLAM/DynSLAM.cpp
    src/DynSLAM/ImageSourceEngine.cpp
    src/DynSLAM/ImageSourceEngine.h
)

add_executable(DynSLAM ${DYNSLAM_SOURCES} ${EXTRA_EXECUTABLE_FLAGS})
target_link_libraries(DynSLAM ${OpenCV_LIBS})
# TODO(andrei): Rename this to ITMEngine or something for clarity.
target_link_libraries(DynSLAM Engine)
# TODO(andrei): Ditto. These are the InfiniTAM utilities!
target_link_libraries(DynSLAM Utils)
target_link_libraries(DynSLAM InstRecLib)

IF(WITH_BACKWARDS_CPP)
  # This will add libraries, definitions and include directories needed by backward
  # by setting each property on the target.
  add_backward(DynSLAM)
ENDIF()

