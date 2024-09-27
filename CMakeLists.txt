# Does your system only have an older version of CMake? Not to worry!
# CMake offers download-and-use binary packages, with no installation 
# necessary...  Visit https://cmake.org/download/ and grab one for 
# your platform. They are not finicky with library dependencies, so
# compatability is very likely. Also, the package's CMake binary will
# not mistake any other local CMake-related files for its own.
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

PROJECT(cuda-api-wrappers
	VERSION 0.8.0
	DESCRIPTION "Thin C++-flavored wrappers for the CUDA Runtime API"
	HOMEPAGE_URL https://github.com/eyalroz/cuda-api-wrappers
)

include(GNUInstallDirs)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib/")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib/")

find_package(CUDAToolkit 9.0 REQUIRED)
enable_language(CXX) # required for using Threads
find_package(Threads REQUIRED)
include(CheckLibraryExists)
check_library_exists(m pow "" libm_exists)
if(libm_exists)
	set(c_math_lib m)
endif()

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.4)
	foreach(tgt in nvfatbin nvfatbin_static)
		if (NOT TARGET ${tgt})
			_CUDAToolkit_find_and_add_import_lib(${tgt})
		endif()
	endforeach()
endif()
if (NOT TARGET CUDA::cufilt)
	_CUDAToolkit_find_and_add_import_lib(cufilt)
endif()

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)



set(targets runtime-and-driver nvtx rtc fatbin)
set(prefixed-targets "")
set(caw_namespace "cuda-api-wrappers")

foreach(wrapper_lib ${targets})
	# First ugly hack to facilitate FetchContent use:
	# Prefix target names with something project-specific
	set(caw_lib "caw_${wrapper_lib}")
	add_library(${caw_lib} INTERFACE)
	target_compile_features(${caw_lib} INTERFACE cxx_std_11) # This means _at least_ C++11
	target_include_directories(
		${caw_lib}
		INTERFACE
		"$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
		"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
	)
	target_link_libraries(${caw_lib} INTERFACE CUDA::cudart CUDA::cuda_driver)

	# These next three dependencies should be carried by the CUDA libraries themselves...
	# but they aren't - this is CMake bug 25665
	target_link_libraries(${caw_lib} INTERFACE Threads::Threads ${CMAKE_DL_LIBS} ${c_math_lib} )
	if(UNIX AND NOT APPLE)
		target_link_libraries(${caw_lib} INTERFACE rt)
	endif()

	# Targets using these libraries should be compiled with C++11 _at least_,
	# but it doesn't seem there's a way to express that at the CMake level

	# Additional hacks for facilitating FetchContent use,
	# which in particular will let you use the same target names as if you had
	# imported them with find_package
	add_library("${caw_namespace}::${wrapper_lib}" ALIAS ${caw_lib})
	list(APPEND prefixed-targets ${caw_lib})
	set_target_properties(${caw_lib}
		PROPERTIES
		EXPORT_NAME ${wrapper_lib}
		OUTPUT_NAME ${wrapper_lib}
		WINDOWS_EXPORT_ALL_SYMBOLS ON
	)
endforeach()
add_library("cuda-api-wrappers::driver-and-runtime" ALIAS caw_runtime-and-driver)
target_link_libraries(caw_rtc INTERFACE cuda-api-wrappers::runtime-and-driver CUDA::nvrtc)
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.1)
	if (TARGET CUDA::nvptxcompiler)
		target_link_libraries(caw_rtc INTERFACE CUDA::nvptxcompiler)
		set(ptx_compiler_target nvptxcompiler)
	elseif (TARGET CUDA::nvptxcompiler_static)
		target_link_libraries(caw_rtc INTERFACE CUDA::nvptxcompiler_static)
	else()
		message(WARNING "No valid NVIDIA PTX Compiler target is available")
	endif()
endif()
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.4)
	if (TARGET CUDA::nvfatbin)
		target_link_libraries(caw_fatbin INTERFACE CUDA::nvfatbin)
	elseif (TARGET CUDA::nvfatbin)
		target_link_libraries(caw_fatbin INTERFACE CUDA::nvfatbin_static)
	elseif(EXISTS "${CUDA_nvfatbin_LIBRARY}") 
		target_link_libraries(caw_fatbin INTERFACE "${CUDA_nvfatbin_LIBRARY}")
	elseif(EXISTS "${CUDA_nvfatbin_static_LIBRARY}") 
		target_link_libraries(caw_fatbin INTERFACE "${CUDA_nvfatbin_static_LIBRARY}")
	else()
		message(WARNING "Could not locate a valid NVIDIA fatbin creator target or library file")
	endif()

	if (TARGET CUDA::cufilt)
		target_link_libraries(caw_runtime-and-driver INTERFACE CUDA::cufilt)
	elseif(EXISTS "${CUDA_cufilt_LIBRARY}")
		target_link_libraries(caw_runtime-and-driver INTERFACE "${CUDA_cufilt_LIBRARY}")
	else()
		message(WARNING "Could not locate the cufilt demangling library")
	endif()

endif()

target_link_libraries(caw_fatbin INTERFACE cuda-api-wrappers::runtime-and-driver)
target_link_libraries(caw_nvtx INTERFACE cuda-api-wrappers::runtime-and-driver)

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 10.0)
	target_link_libraries(caw_nvtx INTERFACE CUDA::nvtx3)
else()
	target_link_libraries(caw_nvtx INTERFACE CUDA::nvToolsExt)
endif()


# Note: This is a bit like a poor man's configure.h file;
# but for two settings I won't bother creating one of those
if(DEFINED CMAKE_USE_PTHREADS_INIT)
	target_compile_definitions(caw_nvtx INTERFACE "" "CUDA_API_WRAPPERS_USE_PTHREADS")
	target_compile_definitions(caw_fatbin INTERFACE "" "CUDA_API_WRAPPERS_USE_PTHREADS")
elseif(DEFINED CMAKE_USE_WIN32_THREADS_INIT)
	target_compile_definitions(caw_nvtx INTERFACE "" "CUDA_API_WRAPPERS_USE_WIN32_THREADS")
	target_compile_definitions(caw_fatbin INTERFACE "" "CUDA_API_WRAPPERS_USE_WIN32_THREADS")
endif()

# --------
# Examples
# --------

option(CAW_BUILD_EXAMPLES "Build example programs" OFF)

if (CAW_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

# ------------------------
# Installing the libraries
# ------------------------

install(FILES "${PROJECT_SOURCE_DIR}/cmake/cuda-api-wrappers-config.cmake"
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cuda-api-wrappers")

install(
	TARGETS ${prefixed-targets}
	EXPORT cuda-api-wrappers_export
	RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
	ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

export(
	EXPORT cuda-api-wrappers_export
	NAMESPACE "${caw_namespace}::"
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
	NAMESPACE "${caw_namespace}::"
	FILE "cuda-api-wrappers-targets.cmake"
)

include(CMakePackageConfigHelpers)

# The SameMinorVersion parameter requires CMake 3.11.
# If not supported, fall back to SameMajorVersion.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.11)
	set(COMPAT_SETTING SameMinorVersion)
else()
	set(COMPAT_SETTING SameMajorVersion)
endif()
write_basic_package_version_file(
	"cuda-api-wrappers-config-version.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY ${COMPAT_SETTING}
)


install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/cuda-api-wrappers-config-version.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cuda-api-wrappers"
)
