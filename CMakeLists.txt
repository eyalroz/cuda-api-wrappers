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
	VERSION 0.3.2
	HOMEPAGE_URL https://github.com/eyalroz/cuda-api-wrappers
	LANGUAGES CUDA CXX)

find_package(Threads REQUIRED)

include(FindCUDA)
include(GNUInstallDirs)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib/")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib/")

add_library(cuda-api-wrappers
	src/cuda/api/profiling.cpp
)

set_property(TARGET cuda-api-wrappers PROPERTY CXX_STANDARD 11)
set_property(TARGET cuda-api-wrappers PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET cuda-api-wrappers PROPERTY CXX_EXTENSIONS OFF)

# Note: This is a bit like a poor man's configure.h file;
# but for two settings I won't bother creating one of those
if(DEFINED CMAKE_USE_PTHREADS_INIT)
	target_compile_definitions(cuda-api-wrappers PUBLIC "" "CUDA_API_WRAPPERS_USE_PTHREADS")
elseif(DEFINED CMAKE_USE_WIN32_THREADS_INIT)
	target_compile_definitions(cuda-api-wrappers PUBLIC "" "CUDA_API_WRAPPERS_USE_WIN32_THREADS")
endif()

target_include_directories(
	cuda-api-wrappers
	PUBLIC
	"$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
	"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# The following lines _should_ have been something like:
# target_link_libraries(cuda-api-wrappers PUBLIC CUDA::CUDALibs Threads::Threads)
# ... but that's not supported.
target_include_directories(cuda-api-wrappers PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda-api-wrappers
	PUBLIC ${CUDA_LIBRARIES}
	PRIVATE Threads::Threads)

option(BUILD_EXAMPLES "Build example programs" OFF)

if (BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

configure_file("${PROJECT_SOURCE_DIR}/cmake/cuda-api-wrappers-config.cmake.in"
	"${PROJECT_BINARY_DIR}/cuda-api-wrappers-config.cmake" @ONLY)

install(FILES "${PROJECT_BINARY_DIR}/cuda-api-wrappers-config.cmake"
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cuda-api-wrappers")

install(
	TARGETS cuda-api-wrappers
	EXPORT cuda-api-wrappers_export
	RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
	ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# If you want to use the library without installing it, you'll need to
# copy the file we generate here, cuda-api-wrappers-targets.cmake, into the
# build directory, and set `cuda-api-wrappers_DIR` to that build directory
# in your own CMake project.
export(
	EXPORT cuda-api-wrappers_export
	NAMESPACE "cuda-api-wrappers::"
	FILE "${PROJECT_BINARY_DIR}/cuda-api-wrappers-targets.cmake"
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
	FILE "cuda-api-wrappers-targets.cmake"
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
	"cuda-api-wrappers-config-version.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMinorVersion
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/cuda-api-wrappers-config-version.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cuda-api-wrappers"
)
