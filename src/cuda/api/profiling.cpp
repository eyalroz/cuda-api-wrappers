#include <cuda/api/profiling.h>
#include <cuda/api/error.hpp>

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>

#include <mutex>

namespace cuda {
namespace profiling {

namespace mark {

namespace detail {
static std::mutex profiler_mutex; // To prevent multiple threads from accessing the profiler simultaneously
}

void point(const std::string& description, color_t color)
{
	std::lock_guard<std::mutex> { detail::profiler_mutex };
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
	const std::string& description, ::cuda::profiling::range::Type type, color_t color)
{
	std::lock_guard<std::mutex> { detail::profiler_mutex };
	nvtxEventAttributes_t range_attributes;
	range_attributes.version   = NVTX_VERSION;
	range_attributes.size      = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	range_attributes.colorType = NVTX_COLOR_ARGB;
	range_attributes.color     = color;
	range_attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
	range_attributes.message.ascii = description.c_str();
	nvtxRangeId_t range_id = nvtxRangeStartEx(&range_attributes);
	static_assert(sizeof(range::handle_t) == sizeof(nvtxRangeId_t),
		"Can't use range::handle_t as a cover for nvtxRangeId_t - they're not the same size");
	return reinterpret_cast<range::handle_t>(range_id);
}

void range_end(range::handle_t range)
{
	nvtxRangeEnd(reinterpret_cast<nvtxRangeId_t>(range));
}

} // namespace mark


scoped_range_marker::scoped_range_marker(const std::string& description, profiling::range::Type type)
{
	range = profiling::mark::range_start(description);
}

scoped_range_marker::~scoped_range_marker()
{
	// TODO: Can we check the range for validity somehow?
	profiling::mark::range_end(range);
}

__host__ void start()
{
	auto status = cudaProfilerStart();
	throw_if_error(status, "Starting to profile");
}

__host__ void stop()
{
	auto status = cudaProfilerStop();
	throw_if_error(status, "Starting to profile");
}

void name_host_thread(pthread_t thread_id, const std::string& name)
{
	nvtxNameOsThreadA(thread_id, name.c_str());
}

void name_host_thread(pthread_t thread_id, const std::wstring& name)
{
	nvtxNameOsThreadW(thread_id, name.c_str());
}

void name_this_thread(const std::string& name)
{
	name_host_thread(pthread_self(), name);
}


} // namespace profiling
} // namespace cuda

