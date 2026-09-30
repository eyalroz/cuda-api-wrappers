#include <cuda/api.hpp>
#if !defined(_MSC_VER) || CUDA_VERSION >= 12000
// MSVC + CMake on Windows has trouble with NVTX header location
#include <cuda/nvtx.hpp>
#endif
#include <cuda/rtc.hpp>
#include <cuda/fatbin.hpp>
#include <cuda/nvtx.hpp>

cuda_::device::id_t get_current_device_id()
{
	auto device = cuda_::device::current::get();
	return device.id();
}

cuda_::fatbin_builder_t make_fatbin_builder	()
{
	return cuda_::fatbin_builder::create({});
}

cuda_::rtc::compilation_options_t<cuda_::cuda_cpp> make_cpp_compilation_options()
{
	return {};
}

void name_magic()
{
	cuda_::profiling::name_this_thread("magic thread");
}
