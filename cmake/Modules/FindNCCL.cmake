#
# Find the nccl libraries
#
# Originally based on PyTorch's NCCL find module, see:
# https://github.com/pytorch/
# and the license and copyright notice at:
# https://github.com/pytorch/pytorch/blob/master/LICENSE
# 
# Modifications (C) Eyal Rozenberg, 2021
# (and there are many of those)
#
# The following environment variables are optionally searched for defaults
#  NCCL_ROOT: Base directory where all NCCL components are found
#  NCCL_INCLUDE_DIR: Directory where NCCL header is found
#  NCCL_LIB_DIR: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIR
#  NCCL_LIBRARIES
#  NCCL_LIBRARY_DIRS
#  NCCL_VERSION
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install NCCL in the same location as the CUDA toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

if (NOT NCCL_INCLUDE_DIR)
	# Note: Assuming CMake >= 3.12, so $ENC{NCCL_ROOT} should be searched automatically
	find_path(NCCL_INCLUDE_DIR
		NAMES nccl.h
		HINTS 
		$ENV{NCCL_DIR}/include
		$ENV{NCCL_INCLUDE_DIR}
		${CUDAToolkit_INCLUDE_DIRS}
		)
endif()
set(NCCL_INCLUDE_DIR ${NCCL_INCLUDE_DIR} CACHE STRING "NVIDIA NCCL library include directories")
mark_as_advanced(NCCL_INCLUDE_DIR)


# TODO: THis part is non-portable
if (USE_STATIC_NCCL)
	MESSAGE(STATUS "USE_STATIC_NCCL is set. Linking with static NCCL library.")
	SET(NCCL_LIBNAME "nccl_static")
	if (NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
		set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
	endif()
else()
	SET(NCCL_LIBNAME "nccl")
	if (NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
		set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
	endif()
endif()

set(NCCL_LIBDIR_HINTS $ENV{NCCL_DIR}/lib)
foreach(nccl_lib_path ${NCCL_LIBRARIES})
	get_filename_component(HINT ${nccl_lib_path} DIRECTORY)
	list(APPEND NCCL_LIBDIR_HINTS ${HINT})
endforeach()

if (NOT NCCL_LIBRARIES)
	find_library(NCCL_LIBRARIES
		NAMES ${NCCL_LIBNAME}
		HINTS 
			${NCCL_LIBDIR_HINTS}
			$ENV{NCCL_LIB_DIR}
		)
endif()
set(NCCL_LIBRARIES ${NCCL_LIBRARIES} CACHE STRING "NVIDIA NCCL library objects" FORCE)
mark_as_advanced(NCCL_LIBRARIES)

set(NCCL_VERSION $ENV{NCCL_VERSION} CACHE STRING "Version of NCCL to build with")
mark_as_advanced(NCCL_VERSION)

if(NCCL_INCLUDE_DIR AND NCCL_LIBRARIES AND NOT NCCL_VERSION)  # obtaining NCCL version and some sanity checks
	set(NCCL_HEADER_FILE "${NCCL_INCLUDE_DIR}/nccl.h")
	set(OLD_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
	list(APPEND CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
	include(CheckCXXSymbolExists)
	check_cxx_symbol_exists(NCCL_VERSION_CODE "${NCCL_HEADER_FILE}" NCCL_VERSION_DEFINED)

	if (NCCL_VERSION_DEFINED)
		set(version_getter_src_file "${PROJECT_BINARY_DIR}/detect_nccl_version.cc")
		list(APPEND CMAKE_REQUIRED_INCLUDES ${NCCL_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
		file(WRITE "${version_getter_src_file}" "
			#include <iostream>
			#include <nccl.h>
			int main()
			{
				std::cout << NCCL_MAJOR << '.' << NCCL_MINOR << '.' << NCCL_PATCH;

				int x;
				ncclGetVersion(&x);
				return x == NCCL_VERSION_CODE;
			}
		")
		try_run(version_getter_run_succeeded version_getter_compile_succeeded ${PROJECT_BINARY_DIR} ${version_getter_src_file}
			RUN_OUTPUT_VARIABLE NCCL_VERSION_FROM_RUN
				  CMAKE_FLAGS  "-DINCLUDE_DIRECTORIES=${CMAKE_REQUIRED_INCLUDES}"
				  LINK_LIBRARIES ${NCCL_LIBRARIES})
		if (NOT version_getter_compile_succeeded)
			message(FATAL_ERROR "Could not compile a program for reporting the NCCL version")
		endif()
		if (NOT version_getter_run_succeeded)
			message(FATAL_ERROR "NCCL version getter program failed")
		endif()
		set(NCCL_VERSION "${NCCL_VERSION_FROM_RUN}" CACHE STRING "Version of NCCL to build with" FORCE)
	else()
		message(WARNING "NCCL version: < 2.3.5-5 (no NCCL_VERSION_CODE)")
	endif ()
	set (CMAKE_REQUIRED_INCLUDES ${OLD_CMAKE_REQUIRED_INCLUDES})
endif()

if(NCCL_LIBRARIES AND NCCL_INCLUDE_DIR)
	add_library(NCCL::NCCL INTERFACE IMPORTED)

	set_target_properties(NCCL::NCCL 
		PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES ${NCCL_INCLUDE_DIR}
		INTERFACE_LINK_LIBRARIES ${NCCL_LIBRARIES}
	)
	if (NCCL_VERSION)
		set_target_properties(NCCL::NCCL 
			PROPERTIES
			VERSION ${NCCL_VERSION}
		)
	endif()
endif()

include(FindPackageHandleStandardArgs)
#if (NCCL_VERSION)
	find_package_handle_standard_args(NCCL 
		REQUIRED_VARS NCCL_LIBRARIES NCCL_INCLUDE_DIR 
	    VERSION_VAR NCCL_VERSION
	)
#else()
#	find_package_handle_standard_args(NCCL 
#		REQUIRED_VARS NCCL_LIBRARIES NCCL_INCLUDE_DIR 
#	)
#endif()
