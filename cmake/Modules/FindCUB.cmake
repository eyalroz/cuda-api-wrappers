# Search for an installation of CUB, looking for "cub/cub.cuh".
# Then, extract the CUB version from its CHANGE_LOG.TXT file.
#
# Two directory structures are supported.
# The first from extracting CUB directly somewhere:
#   .../cub/cub.h
#   .../CHANGE_LOG.TXT
#
# The second from packaging the include files under e.g. /usr/local/include/,
# and the rest under e.g. /usr/local/share/:
#   .../include/cub/cub.h
#   .../share/cub/CHANGE_LOG.TXT

find_path(
  CUB_INCLUDE_DIR
  cub/cub.cuh
  HINTS
      "${CMAKE_SOURCE_DIR}"
      "${CMAKE_SOURCE_DIR}/include"
      "${PROJECT_SOURCE_DIR}"
      "${PROJECT_SOURCE_DIR}/include"
      ENV CUB_DIR
      ENV CUB_INCLUDE_DIR
      ENV CUB_PATH
  DOC "NVIDIA CUB include directory"
)

if(EXISTS "${CUB_INCLUDE_DIR}")
  set(CUB_FOUND 1)
  # look for CHANGE_LOG.TXT
  if(EXISTS "${CUB_INCLUDE_DIR}/CHANGE_LOG.TXT")
    file(READ "${CUB_INCLUDE_DIR}/CHANGE_LOG.TXT" CHANGELOG)
  elseif(EXISTS "${CUB_INCLUDE_DIR}/../share/cub/CHANGE_LOG.TXT")
    file(READ "${CUB_INCLUDE_DIR}/../share/cub/CHANGE_LOG.TXT" CHANGELOG)
  endif()
  # read the version from CHANGE_LOG.TXT, or set it to "unknown"
  if(CHANGELOG)
    string(REGEX MATCH "([0-9]+\.[0-9]+\.[0-9]+)" CUB_VERSION "${CHANGELOG}")
  else()
    set(CUB_VERSION "unknown")
  endif()
  message(STATUS "Found CUB: ${CUB_INCLUDE_DIR} (version \"${CUB_VERSION}\", minimum required is \"${CUB_FIND_VERSION}\")")
else()
  set(CUB_FOUND 0)
endif()
