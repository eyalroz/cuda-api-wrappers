/**
 * @file
 *
* @brief Contains @ref fatbin_builder::options_t class and related definitions
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_FATBIN_BUILDER_OPTIONS_HPP_
#define CUDA_API_WRAPPERS_FATBIN_BUILDER_OPTIONS_HPP_

#include "../api/device_properties.hpp"
#include "../api/detail/option_marshalling.hpp"
#include "../api/types.hpp"

#include <array>
#include <sstream>

namespace cuda {

///@cond
class module_t;
///@endcond

namespace fatbin_builder {


/*

Fatbin options (not including deprecated ones):

 -compress=<bool> Enable (true) / disable (false) compression (default: true).

 -compress-all Compress everything in the fatbin, even if it’s small.

 -cuda Specify CUDA (rather than OpenCL).
 -opencl Specify OpenCL (rather than CUDA).
 -host=<name>   Specify host operating system. Valid options are “linux”, “windows” (“mac”  is deprecated)

 -g Generate debug information.

*/

struct options_t final {

	enum : bool {
		width_32_bits = false, width_64_bits = true
	};
	optional<bool> use_64_bit_entry_width{width_64_bits};

	enum : bool {
		dont_compress = false, do_compress = true
	};
	optional<bool> compress{do_compress};

	enum : bool {
		only_compress_large_objects = false, compression_for_everything = true
	};
	optional<bool> apply_compression_to_small_objects{only_compress_large_objects};

	enum ecosystem_t {
		cuda, opencl
	};
	optional<ecosystem_t> ecosystem;

	enum host_os_t {
		windows, linux
	};
	optional<host_os_t> targeted_host_os;
};

namespace detail_ {

struct marshalled_options_t {
	::std::size_t num_options;
	::std::string option_str;
};

} // namespace detail

} // namespace fatbin_builder

namespace marshalling {

namespace detail_ {

template <typename MarshalTarget, typename Delimiter>
struct gadget<fatbin_builder::options_t, MarshalTarget, Delimiter> {
	static void process(
		const fatbin_builder::options_t &opts,
		MarshalTarget &marshalled, Delimiter delimiter,
		bool need_delimiter_after_last_option)
	{
		using fatbin_builder::options_t;
		opt_start_t<Delimiter> opt_start { delimiter };
		if (opts.use_64_bit_entry_width) {
			marshalled << opt_start << '-' << (opts.use_64_bit_entry_width.value() ? "64" : "32");
		}
		if (opts.compress) {
			marshalled << opt_start << "-compress=" << (opts.compress.value() ? "true" : "false");
		}
		if (opts.apply_compression_to_small_objects.value_or(false)) {
			marshalled << opt_start << "-compress-all";
		}
		if (opts.ecosystem) {
			marshalled << opt_start << '-' << ((opts.ecosystem.value() == options_t::opencl) ? "opencl" : "cuda");
		}
		if (opts.targeted_host_os) {
			marshalled << opt_start << "-host=" << ((opts.targeted_host_os.value() == options_t::windows) ? "windows" : "linux");
		}
		if (need_delimiter_after_last_option) {
			marshalled << opt_start;
		}
	}
};

} // namespace detail_

} // namespace marshalling

} // namespace cuda

#endif // CUDA_API_WRAPPERS_FATBIN_BUILDER_OPTIONS_HPP_
