include(CMakeFindDependencyMacro)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_dependency(Threads)
find_dependency(CUDAToolkit)

include("${CURRENT_LIST_DIR}/cuda-api-wrappers-targets.cmake")
