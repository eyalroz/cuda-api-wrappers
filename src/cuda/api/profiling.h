/**
 * @file profiling.h
 *
 * @brief wrappers for the CUDA profiler API functions,
 * and convenience RAII classes for profiler-output-marked
 * time range and points.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_PROFILING_H_
#define CUDA_API_WRAPPERS_PROFILING_H_

#include <cuda/api/types.h>
#include <pthread.h>

#include <mutex>

namespace cuda {

namespace profiling {

struct color_t {
	using underlying_type = uint32_t;
	unsigned char alpha, red, green, blue;

	static constexpr color_t from_hex(underlying_type raw_argb) {
		return {
			(unsigned char) ((raw_argb >> 24) & 0xFF),
			(unsigned char) ((raw_argb >> 16) & 0xFF),
			(unsigned char) ((raw_argb >>  8) & 0xFF),
			(unsigned char) ((raw_argb >>  0) & 0xFF),
		};
	}
	operator underlying_type()	{ return as_hex(); }
	underlying_type as_hex()
	{
		return
			((underlying_type) alpha)  << 24 |
			((underlying_type) red  )  << 16 |
			((underlying_type) green)  <<  8 |
			((underlying_type) blue )  <<  0;
	}
	static constexpr color_t Black()       { return from_hex(0x00000000); }
	static constexpr color_t White()       { return from_hex(0x00FFFFFF); }
	static constexpr color_t FullRed()     { return from_hex(0x00FF0000); }
	static constexpr color_t FullGreen()   { return from_hex(0x0000FF00); }
	static constexpr color_t FullBlue()    { return from_hex(0x000000FF); }
	static constexpr color_t FullYellow()  { return from_hex(0x00FFFF00); }
	static constexpr color_t LightRed()    { return from_hex(0x00FFDDDD); }
	static constexpr color_t LightGreen()  { return from_hex(0x00DDFFDD); }
	static constexpr color_t LightBlue()   { return from_hex(0x00DDDDFF); }
	static constexpr color_t LightYellow() { return from_hex(0x00FFFFDD); }
	static constexpr color_t DarkRed()     { return from_hex(0x00880000); }
	static constexpr color_t DarkGreen()   { return from_hex(0x00008800); }
	static constexpr color_t DarkBlue()    { return from_hex(0x00000088); }
	static constexpr color_t DarkYellow()  { return from_hex(0x00888800); }
};

namespace range {
enum class Type { unspecified, Kernel, pci_express_transfer	};
using handle_t = uint64_t;
} // namespace range

namespace mark {
void            point (const std::string& message, color_t color = color_t::Black());
range::handle_t range_start (
	const std::string& description, range::Type type = range::Type::unspecified, color_t color = color_t::LightRed());
void            range_end (range::handle_t range);
} // namespace mark

/**
 * A RAII class whose scope of existence is reflected as a range in the profiler.
 * Use it in the scope in which you perform some interesting operation, e.g.
 * perform a synchronous I/O operation (and have it conclude of course), or
 * launch and synch several related kernels.
 */
class scoped_range_marker {
public:
	scoped_range_marker(
		const std::string& description,
		profiling::range::Type type = profiling::range::Type::unspecified);
	~scoped_range_marker();
protected:
	profiling::range::handle_t range;
};

/**
 * Start CUDA profiling for the current process
 */
void start();

/**
 * Sttop CUDA profiling for the current process
 */
void stop();

/**
 * A class to instantiate in the part of your application
 * which does any work you intend to use the CUDA profiler
 * to profile. This could well be your main() function.
 */
class scope {
public:
	scope() { start(); }
	~scope() { stop(); }
};

namespace naming {

/**
 * Have the profiler refer to a host OS thread using
 * a specified string identifier (rather than its
 * hard-to-decipher (alpha)numeric ID).
 */
void name_host_thread(pthread_t thread_id, const std::string&);
void name_host_thread(pthread_t thread_id, const std::wstring&);
void name_this_thread(const std::string&);

//void name_device_stream(device::id_t  device, stream::id_t stream);

} // namespace naming

} // namespace profiling

} // namespace cuda

#endif /* CUDA_API_WRAPPERS_PROFILING_H_ */
