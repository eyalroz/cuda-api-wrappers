#include <cuda/nvtx/profiling.hpp>
#include <cuda_profiler_api.h>

#include <cuda/api/error.hpp>
#if CUDART_VERSION >= 1000 && defined(_WIN32)
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#else
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>
#endif

#include <mutex>

#ifdef CUDA_API_WRAPPERS_USE_PTHREADS
#include <pthread.h>
#else
#ifdef CUDA_API_WRAPPERS_USE_WIN32_THREADS
#include <processthreadsapi.h>
#endif
#endif

namespace cuda {
namespace profiling {

namespace mark {

namespace detail_ {
static ::std::mutex profiler_mutex; // To prevent multiple threads from accessing the profiler simultaneously
}

void point(const ::std::string& description, color_t color)
{
	::std::lock_guard<::std::mutex> { detail_::profiler_mutex };
	// logging?
	nvtxEventAttributes_t eventAttrib = {0};
	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = color;
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	eventAttrib.message.ascii = description.c_str();
	nvtxMarkEx(&eventAttrib);
}

range::handle_t range_start(
	const ::std::string& description, ::cuda::profiling::range::type_t type, color_t color)
{
	(void) type; // Currently not doing anything with the type; maybe in the future
	::std::lock_guard<::std::mutex> { detail_::profiler_mutex };
	nvtxEventAttributes_t range_attributes;
	range_attributes.version   = NVTX_VERSION;
	range_attributes.size      = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	range_attributes.colorType = NVTX_COLOR_ARGB;
	range_attributes.color     = color;
	range_attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
	range_attributes.message.ascii = description.c_str();
	nvtxRangeId_t range_handle = nvtxRangeStartEx(&range_attributes);
	static_assert(::std::is_same<range::handle_t, nvtxRangeId_t>::value,
		"range::handle_t must be the same type as nvtxRangeId_t - but isn't.");
	return range_handle;
}

void range_end(range::handle_t range_handle)
{
	static_assert(::std::is_same<range::handle_t, nvtxRangeId_t>::value,
		"range::handle_t must be the same type as nvtxRangeId_t - but isn't.");
	nvtxRangeEnd(range_handle);
}

} // namespace mark


scoped_range_marker::scoped_range_marker(const ::std::string& description, profiling::range::type_t type)
{
	range = profiling::mark::range_start(description, type);
}

scoped_range_marker::~scoped_range_marker()
{
	// TODO: Can we check the range for validity somehow?
	profiling::mark::range_end(range);
}

void start()
{
	auto status = cudaProfilerStart();
	throw_if_error(status, "Starting to profile");
}

void stop()
{
	auto status = cudaProfilerStop();
	throw_if_error(status, "Starting to profile");
}

namespace naming {

template <>
void name_host_thread<char>(uint32_t thread_id, const ::std::string& name)
{
	nvtxNameOsThreadA(thread_id, name.c_str());
}

template <>
void name_host_thread<wchar_t>(uint32_t thread_id, const ::std::wstring& name)
{
	nvtxNameOsThreadW(thread_id, name.c_str());
}

#if defined(CUDA_API_WRAPPERS_USE_WIN32_THREADS) || defined(CUDA_API_WRAPPERS_USE_PTHREADS)

template <typename CharT>
void name_this_thread(const ::std::basic_string<CharT>& name)
{
	auto this_thread_s_native_handle =
#ifdef CUDA_API_WRAPPERS_USE_PTHREADS
		::pthread_self();
#else
		::GetCurrentThreadId();
#endif
	name_host_thread<CharT>(this_thread_s_native_handle, name);
}

#endif // defined(CUDA_API_WRAPPERS_USE_WIN32_THREADS) || defined(CUDA_API_WRAPPERS_USE_PTHREADS)

} // namespace naming

} // namespace profiling
} // namespace cuda

