# We need version 3.8 for native CUDA support in CMake
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# -----------------------------------
# Project name, version & build type
# -----------------------------------

if(WIN32 AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	# Avoid a build failure when the BUILD_SHARED_LIBS variable is set to true.
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

PROJECT(cuda-api-wrappers
	DESCRIPTION "Thin C++-flavored wrappers for the CUDA Runtime API"
	HOMEPAGE_URL https://github.com/eyalroz/cuda-api-wrappers
	LANGUAGES CUDA CXX)

include(FindCUDA) 
include(GNUInstallDirs)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib/")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib/")

add_library(cuda-api-wrappers
	src/cuda/api/device_properties.cpp
	src/cuda/api/profiling.cpp
)

set_property(TARGET cuda-api-wrappers PROPERTY CXX_STANDARD 11)
set_property(TARGET cuda-api-wrappers PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET cuda-api-wrappers PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(
	cuda-api-wrappers
	PUBLIC
	"$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
	"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# The following lines _should_ have been something like:
# target_link_libraries(cuda-api-wrappers PUBLIC CUDA::CUDALibs)
# ... but that's not supported.
target_include_directories(cuda-api-wrappers PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda-api-wrappers PUBLIC ${CUDA_LIBRARIES})

option(BUILD_EXAMPLES "Build example programs" OFF)

if (BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

install(
	TARGETS cuda-api-wrappers
	EXPORT cuda-api-wrappers_export
	RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
	ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(
	DIRECTORY src/cuda
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
	FILES_MATCHING REGEX "\\.(h|hpp|cuh)$"
)

install(
	EXPORT cuda-api-wrappers_export
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cuda-api-wrappers"
	NAMESPACE "cuda-api-wrappers::"
	# In this CMake config file no dependencies are considered. But since we
	# do not use any `find_package` call here this approach is sufficient.
	FILE cuda-api-wrappers-config.cmake
)

