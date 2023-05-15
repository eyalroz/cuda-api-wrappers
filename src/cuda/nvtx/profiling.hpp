/**
 * @file
 *
 * @brief wrappers for the CUDA profiler API functions,
 * and convenience RAII classes for profiler-output-marked
 * time range and points.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_PROFILING_HPP_
#define CUDA_API_WRAPPERS_PROFILING_HPP_

#include "../api/types.hpp"
#include "../api/error.hpp"
#include "../api/current_context.hpp"
#include "../api/stream.hpp"
#include "../api/event.hpp"
#include "../api/device.hpp"

#include <cudaProfiler.h>

#if CUDA_VERSION >= 10000 || defined(_WIN32)
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCuda.h>
#else
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#endif

#ifdef _WIN32
#include <processthreadsapi.h> // for GetThreadId()
#endif

#ifdef CUDA_API_WRAPPERS_USE_PTHREADS
#include <pthread.h>
#else
#ifdef CUDA_API_WRAPPERS_USE_WIN32_THREADS
#include <processthreadsapi.h>
#endif
#endif


#include <mutex>
#include <cstdint>
#include <string>
#include <cstdint>
#include <thread>


namespace cuda {

namespace context {

namespace current {

namespace detail_ {

class scoped_existence_ensurer_t;

} // namespace detail_

} // namespace current

} // namespace context

// Note: No implementation for now for nvtxStringHandle_t's

namespace profiling {

namespace detail_ {
inline void set_message(nvtxEventAttributes_t &attrs, const char *c_str) noexcept
{
	attrs.messageType = NVTX_MESSAGE_TYPE_ASCII;
	attrs.message.ascii = c_str;
}

inline void set_message(nvtxEventAttributes_t &attrs, const wchar_t *wc_str) noexcept
{
	attrs.messageType = NVTX_MESSAGE_TYPE_UNICODE;
	attrs.message.unicode = wc_str;
}

inline void set_message(nvtxEventAttributes_t &attrs, nvtxStringHandle_t rsh) noexcept
{
	attrs.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
	attrs.message.registered = rsh;
}

} // namespace detail_

namespace range {

enum class type_t { unspecified, kernel, pci_express_transfer	};

/**
 * The range handle is actually `nvtxRangeId_t`; but - other than this typedef,
 * we don't need to include the nVIDIA Toolkit Extensions headers at all here,
 * and can leave them within the implementation only.
 */
using handle_t = ::std::uint64_t;

} // namespace range

/**
 * @brief An RGB colorspace color value, with potential transparency, which
 * may be used to color elements in timelines or other graphical displays of
 * profiling information.
 */
struct color_t {
	using underlying_type = ::std::uint32_t;
	using channel_value = ::std::uint8_t;
	channel_value alpha, red, green, blue;

	static constexpr color_t from_hex(underlying_type raw_argb) noexcept {
		return {
		(channel_value) ((raw_argb >> 24) & 0xFF),
		(channel_value) ((raw_argb >> 16) & 0xFF),
		(channel_value) ((raw_argb >>  8) & 0xFF),
		(channel_value) ((raw_argb >>  0) & 0xFF),
		};
	}
	operator underlying_type() const noexcept { return as_hex(); }
	underlying_type as_hex() const noexcept
	{
		return
		((underlying_type) alpha)  << 24 |
		((underlying_type) red  )  << 16 |
		((underlying_type) green)  <<  8 |
		((underlying_type) blue )  <<  0;
	}
	static constexpr color_t Black()       noexcept { return from_hex(0x00000000); }
	static constexpr color_t White()       noexcept { return from_hex(0x00FFFFFF); }
	static constexpr color_t FullRed()     noexcept { return from_hex(0x00FF0000); }
	static constexpr color_t FullGreen()   noexcept { return from_hex(0x0000FF00); }
	static constexpr color_t FullBlue()    noexcept { return from_hex(0x000000FF); }
	static constexpr color_t FullYellow()  noexcept { return from_hex(0x00FFFF00); }
	static constexpr color_t LightRed()    noexcept { return from_hex(0x00FFDDDD); }
	static constexpr color_t LightGreen()  noexcept { return from_hex(0x00DDFFDD); }
	static constexpr color_t LightBlue()   noexcept { return from_hex(0x00DDDDFF); }
	static constexpr color_t LightYellow() noexcept { return from_hex(0x00FFFFDD); }
	static constexpr color_t DarkRed()     noexcept { return from_hex(0x00880000); }
	static constexpr color_t DarkGreen()   noexcept { return from_hex(0x00008800); }
	static constexpr color_t DarkBlue()    noexcept { return from_hex(0x00000088); }
	static constexpr color_t DarkYellow()  noexcept { return from_hex(0x00888800); }
};

namespace mark {

namespace detail_ {

// Used to prevent multiple threads from accessing the profiler simultaneously
inline ::std::mutex& get_mutex() noexcept
{
	static ::std::mutex profiler_mutex;
	return profiler_mutex;
}

template <typename CharT>
nvtxEventAttributes_t create_attributes(const CharT* description, color_t color)
{
	nvtxEventAttributes_t eventAttrib = {0};
	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = color;
	profiling::detail_::set_message(eventAttrib,description);
	return eventAttrib;
}

} // namespace detail_

template <typename CharT>
void point(const CharT* description, color_t color)
{
	auto attrs = detail_::create_attributes(description, color);
	::std::lock_guard<::std::mutex> guard{ detail_::get_mutex() };
	// logging?
	nvtxMarkEx(&attrs);
}

template <typename CharT>
range::handle_t range_start(
	const CharT*   description,
	range::type_t  type,
	color_t        color)
{
	(void) type; // Currently not doing anything with the type; maybe in the future
	::std::lock_guard<::std::mutex> guard{ detail_::get_mutex() };
	auto attrs = detail_::create_attributes(description, color);
	nvtxRangeId_t range_handle = nvtxRangeStartEx(&attrs);
	static_assert(::std::is_same<range::handle_t, nvtxRangeId_t>::value,
				  "cuda::profiling::range::handle_t must be the same type as nvtxRangeId_t - but isn't.");
	return range_handle;
}

inline void range_end(range::handle_t range_handle)
{
	static_assert(::std::is_same<range::handle_t, nvtxRangeId_t>::value,
				  "cuda::profiling::range::handle_t must be the same type as nvtxRangeId_t - but isn't.");
	nvtxRangeEnd(range_handle);
}

} // namespace mark

/**
 * Start CUDA profiling for the current process
 */
inline void start()
{
	auto status = cuProfilerStart();
	throw_if_error_lazy(status, "Starting CUDA profiling");
}

/**
 * Stop CUDA profiling for the current process
 */
inline void stop()
{
	auto status = cuProfilerStop();
	throw_if_error_lazy(status, "Stopping CUDA profiling");
}

} // namespace profiling
} // namespace cuda

namespace cuda {

namespace profiling {

namespace mark {

template <typename CharT>
inline void point(const CharT* message)
{
	point(message, color_t::Black());
}

template <typename CharT>
inline range::handle_t range_start(
	const CharT*   description,
	range::type_t  type)
{
	return range_start(description, type, color_t::LightRed());
}

template <typename CharT>
inline range::handle_t range_start(const CharT* description)
{
	return range_start(description, range::type_t::unspecified);
}

} // namespace mark

/**
 * A RAII class whose scope of existence is reflected as a range in the profiler.
 * Use it in the scope in which you perform some interesting operation, e.g.
 * perform a synchronous I/O operation (and have it conclude of course), or
 * launch and synch several related kernels.
 */
class scoped_range_marker {
public:
	template <typename CharT>
	explicit scoped_range_marker(
		const CharT* description,
		profiling::range::type_t type = profiling::range::type_t::unspecified)
	{
		range = profiling::mark::range_start(description, type);
	}

	~scoped_range_marker()
	{
		// TODO: Can we check the range for validity somehow?
		profiling::mark::range_end(range);
	}
protected:
	profiling::range::handle_t range;
};

/**
 * A class to instantiate in the part of your application
 * which does any work you intend to use the CUDA profiler
 * to profile. This could well be your main() function.
 */
class scope {
public:
	scope() { start(); }
	~scope() { stop(); }
protected:
	context::current::detail_::scoped_existence_ensurer_t context_existence_ensurer;
};

#define profile_this_scope() ::cuda::profiling::scope cuda_profiling_scope_{};

namespace detail_ {

template <typename CharT>
void name_host_thread(uint32_t raw_thread_id, const CharT* name);

template <>
inline void name_host_thread<char>(uint32_t raw_thread_id, const char* name)
{
	nvtxNameOsThreadA(raw_thread_id, name);
}

template <>
inline void name_host_thread<wchar_t>(uint32_t raw_thread_id, const wchar_t* name)
{
	nvtxNameOsThreadW(raw_thread_id, name);
}

template <typename CharT>
void name_stream(stream::handle_t stream_handle, const CharT* name);

template <>
inline void name_stream<char>(stream::handle_t stream_handle, const char* name)
{
	nvtxNameCuStreamA(stream_handle, name);
}

template <>
inline void name_stream<wchar_t>(stream::handle_t stream_handle, const wchar_t* name)
{
	nvtxNameCuStreamW(stream_handle, name);
}

template <typename CharT>
inline void name_event(event::handle_t event_handle, const CharT* name);

template <>
inline void name_event<char>(event::handle_t event_handle, const char* name)
{
	nvtxNameCuEventA(event_handle, name);
}

template <>
inline void name_event<wchar_t>(event::handle_t event_handle, const wchar_t* name)
{
	nvtxNameCuEventW(event_handle, name);
}

template <typename CharT>
void name_device(device::id_t device_id, const CharT* name);

template <>
inline void name_device<char>(device::id_t device_id, const char* name)
{
	nvtxNameCuDeviceA(device_id, name);
}

template <>
inline void name_device<wchar_t>(device::id_t device_id, const wchar_t* name)
{
	nvtxNameCuDeviceW(device_id, name);
}

inline void name(::std::thread::id host_thread_id, const char* name)
{
	auto native_handle = *(reinterpret_cast<const ::std::thread::native_handle_type*>(&host_thread_id));
	uint32_t thread_id =
#if _WIN32
		GetThreadId(native_handle);
#else
		native_handle;
#endif
	name_host_thread(thread_id, name);
}

} // namespace detail_

/**
 * @brief Have the profiler refer to the current thread, or another host
 * thread, using a specified string identifier (rather than its numeric ID).
 *
 * @param[in] thread_id  A native numeric ID of the thread; on Linux systems
 * this would be a `pthread_t`, and on Windows - a DWORD (as is returned,
 * for example, by `GetCurrentThreadId()`)
 * @param[in] name The string identifier to use for the specified thread
 */
///@{
template <typename CharT>
void name(const ::std::thread& host_thread, const CharT* name);

template <typename CharT>
void name_this_thread(const CharT* name)
{
	detail_::name(::std::this_thread::get_id(), name);
}
///@}

template <typename CharT>
void name(const stream_t& stream, const CharT* name)
{
	context::current::detail_::scoped_override_t context_setter{stream.context_handle()};
	detail_::name_stream(stream.handle(), name);
}

template <typename CharT>
void name(const event_t& event, const CharT* name)
{
	context::current::detail_::scoped_override_t context_setter{event.context_handle()};
	detail_::name_stream(event.handle(), name);
}

template <typename CharT>
void name(const device_t& device, const CharT* name)
{
	detail_::name_stream(device.id(), name);
}

} // namespace profiling
} // namespace cuda

#endif // CUDA_API_WRAPPERS_PROFILING_HPP_
