include(CMakeFindDependencyMacro)

enable_language(CXX)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_dependency(Threads)
find_dependency(CUDAToolkit REQUIRED)

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.4)
    foreach(tgt nvfatbin nvfatbin_static)
        if (NOT TARGET ${tgt})
            _CUDAToolkit_find_and_add_import_lib(${tgt})
        endif()
    endforeach()
endif()
if (NOT TARGET CUDA::cufilt)
    _CUDAToolkit_find_and_add_import_lib(cufilt)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cuda-api-wrappers-targets.cmake")
