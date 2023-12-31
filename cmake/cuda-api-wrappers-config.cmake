include(CMakeFindDependencyMacro)

enable_language(CXX)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_dependency(Threads)
find_dependency(CUDAToolkit)

include("${CMAKE_CURRENT_LIST_DIR}/cuda-api-wrappers-targets.cmake")
