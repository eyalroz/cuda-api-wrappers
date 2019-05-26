# Find the Modern C++ API wrappers for the CUDA runtime API
#
# The following variables are set if the library is found.
#
#	CUDA_API_WRAPPERS_FOUND
#       True when both the include files and the libraries were found
#
#	CUDA_API_WRAPPERS_INCLUDE_DIR  
#       The path to where the CUB include files are the path which should 
#       be added to a project's include directories to use the API wrappers
#
#	CUDA_API_WRAPPERS_LIB
#       The (static) library file to be linked when using the non-header-only
#       parts of the CUDA API wrappers

find_package(PkgConfig) # will this even help us at all?

find_path(
	CUDA_API_WRAPPERS_INCLUDE_DIR 
	cuda/api_wrappers.hpp
	HINTS
		${CUDA_INCLUDE_DIRS}
		${CMAKE_CURRENT_SOURCE_DIR}/cuda-api-wrappers
		${CMAKE_SOURCE_DIR}
		${PROJECT_SOURCE_DIR}
		/opt 
		/opt/cuda-api-wrappers
		$ENV{HOME}/opt 
		${CMAKE_PREFIX_PATH}/cuda-api-wrappers/lib
		ENV CUDA_API_WRAPPERS_DIR 
		ENV CUDA_API_WRAPPERS_INCLUDE_DIR 
		ENV CUDA_API_WRAPPERS_PATH
	DOC "CUDA Modern C++ API wrappers - include directory"
	PATH_SUFFIXES include src
)
mark_as_advanced(CUDA_API_WRAPPERS_INCLUDE_DIR)

find_library(
	CUDA_API_WRAPPERS_LIB
	HINTS 
		${CMAKE_CURRENT_SOURCE_DIR}/cuda-api-wrappers/lib
		${CMAKE_SOURCE_DIR}
		${PROJECT_SOURCE_DIR}
		ENV CUDA_API_WRAPPERS_DIR 
		ENV CUDA_API_WRAPPERS_LIBRARY_DIR
		ENV CUDA_API_WRAPPERS_INCLUDE_DIR
	NAMES cuda-api-wrappers libcuda-api-wrappers
	PATH_SUFFIXES lib
	DOC "CUDA Modern C++ API wrappers - library"
	NO_DEFAULT_PATH
)
mark_as_advanced(CUDA_API_WRAPPERS_LIB)

# handles the QUIETLY and REQUIRED arguments and sets the find variable to TRUE
# if all of the requirement variables are TRUE. This library is not versioned,
# so we'll not have a version check.
find_package_handle_standard_args(CUDAAPIWrappers
	FOUND_VAR     CUDAAPIWrappers_FOUND
	REQUIRED_VARS CUDA_API_WRAPPERS_INCLUDE_DIR CUDA_API_WRAPPERS_LIB
)

# We don't have any dependent include dirs, so what we want the user code to use is just
# our own directory
set(CUDA_API_WRAPPERS_INCLUDE_DIRS ${CUDA_API_WRAPPERS_INCLUDE_DIR})
# We don't have any dependent libraries, so what we want to user code to use is just
# our own library
set(CUDA_API_WRAPPERS_LIBS "${CUDA_API_WRAPPERS_LIB}")

