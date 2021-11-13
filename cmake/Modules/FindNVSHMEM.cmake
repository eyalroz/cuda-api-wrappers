# Copyright Guillaume Tauzin 2017
# Distributed under the MIT License (MIT)

# .rst:
# FindNVSHMEM
# ------------
#
# Locate the NVSHMEM library
#
#
# Imported targets
# ^^^^^^^^^^^^^^^^
#
# ``NVSHMEM::NVSHMEM``
#
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``NVSHMEM_FOUND``            - Found the NVSHMEM library
# ``NVSHMEM_INCLUDE_DIRS``     - NVSHMEM include directories
# ``NVSHMEM_LIBRARY_DIRS``     - NVSHMEM lib directories
#
#
# Cache variables
# ^^^^^^^^^^^^^^^
#
# The following Cache variables may also be set:
#
# ``NVSHMEM_ROOT``             - The root directory of RapidXml installation
#                                 (may also be set as an environment variable)
#

set(NVSHMEM_INCLUDE_SEARCH_DIRS "")
set(NVSHMEM_LIBRARY_SEARCH_DIRS "")

if(NVSHMEM_ROOT)
  list(APPEND NVSHMEM_INCLUDE_SEARCH_DIRS
              ${NVSHMEM_ROOT}/include
              ${NVSHMEM_ROOT})
  list(APPEND NVSHMEM_LIBRARY_SEARCH_DIRS
              ${NVSHMEM_ROOT}/lib
              ${NVSHMEM_ROOT})
elseif(DEFINED ENV{NVSHMEM_ROOT})
  set(NVSHMEM_ROOT $ENV{NVSHMEM_ROOT})
  list(APPEND NVSHMEM_INCLUDE_SEARCH_DIRS
              $ENV{NVSHMEM_ROOT}/include
              $ENV{NVSHMEM_ROOT})
  list(APPEND NVSHMEM_LIBRARY_SEARCH_DIRS
              $ENV{NVSHMEM_ROOT}/lib
              $ENV{NVSHMEM_ROOT})
endif()

set(NVSHMEM_KNOWN_VERSIONS "0.1.0")

set(NVSHMEM_PATH_SUFFIXES)
foreach(NVSHMEM_VERSION ${NVSHMEM_KNOWN_VERSIONS})
  list(APPEND NVSHMEM_PATH_SUFFIXES "nvshmem-${NVSHMEM_VERSION}")
endforeach()

find_path(
    NVSHMEM_INCLUDE_DIRS
    NAMES         shmem.h shmemx.h
    HINTS         ${NVSHMEM_INCLUDE_SEARCH_DIRS}
    PATH_SUFFIXES ${NVSHMEM_PATH_SUFFIXES})

find_library(NVSHMEM_LIBRARY_DIRS
    NAMES         shmem
    PATHS         ${NVSHMEM_LIBRARY_SEARCH_DIRS}
    PATH_SUFFIXES ${NVSHMEM_PATH_SUFFIXES}
    NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(NVSHMEM DEFAULT_MSG NVSHMEM_INCLUDE_DIRS NVSHMEM_LIBRARY_DIRS)

mark_as_advanced(NVSHMEM_INCLUDE_DIRS NVSHMEM_LIBRARY_DIRS)

if(NVSHMEM_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -L${NVSHMEM_ROOT}/lib -lshmem -lcuda" )
#  add_library(NVSHMEM::NVSHMEM INTERFACE IMPORTED)
#  set_property(TARGET NVSHMEM::NVSHMEM PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${NVSHMEM_INCLUDE_DIRS})
#  set_property(TARGET NVSHMEM::NVSHMEM PROPERTY INTERFACE_LINK_LIBRARIES "${NVSHMEM_LIBRARY_DIRS} -lshmem")
endif()

