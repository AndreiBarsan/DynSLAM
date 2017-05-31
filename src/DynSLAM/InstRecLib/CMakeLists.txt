#############################
# Specify the source files  #
#############################

FILE(GLOB INSTRECLIB_SOURCES *.cpp)
FILE(GLOB INSTRECLIB_HEADERS *.h)
FILE(GLOB INSTRECLIB_UTILS_SOURCES Utils/*.cpp)
FILE(GLOB INSTRECLIB_UTILS_HEADERS Utils/*.h)

# TODO(andrei): This seems to error out on a fresh setup; we may need a better system.
# Once we build pangolin the first time, the problem goes away, but that's not a very
# healthy approach...
#find_package(Pangolin REQUIRED)
#include_directories(${Pangolin_INCLUDE_DIRS})

IF(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
ELSE()
  # We could still run this without CUDA if using, e.g., pre-segmented frames.
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
ENDIF()

##############################################################
# Specify the include directories, target and link libraries #
##############################################################

# TODO(andrei): In the future, this is where you would add dependencies on Caffe or whatever.

add_library(InstRecLib
${INSTRECLIB_SOURCES}
${INSTRECLIB_UTILS_SOURCES}
${INSTRECLIB_HEADERS}
${INSTRECLIB_UTILS_HEADERS})
