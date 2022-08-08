/**
 * @file
 *
 * @brief Contains the @ref cuda::rtc::program_t class and related code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NVRTC_PROGRAM_HPP_
#define CUDA_API_WRAPPERS_NVRTC_PROGRAM_HPP_

#include <cuda/nvrtc/compilation_options.hpp>
#include <cuda/nvrtc/compilation_output.hpp>
#include <cuda/nvrtc/error.hpp>
#include <cuda/nvrtc/types.hpp>
#include <cuda/api.hpp>

#include <vector>
#include <iostream>

namespace cuda {

namespace rtc {

using const_cstrings_span = span<const char* const>;
using const_cstring_pairs_span = span<::std::pair<const char* const, const char* const>>;

class program_t;


namespace detail_ {

::std::string identify(const program_t &program);

} // namespace detail_

namespace program {

namespace detail_ {

inline program::handle_t create(
	const char *program_name,
	const char *program_source,
	int num_headers,
	const char *const *header_sources,
	const char *const *header_names)
{
	program::handle_t program_handle;
	auto status = nvrtcCreateProgram(
		&program_handle, program_source, program_name, (int) num_headers, header_sources, header_names);
	throw_if_error(status, "Failed creating an NVRTC program (named " + ::std::string(program_name) + ')');
	return program_handle;
}

inline void register_global(handle_t program_handle, const char *global_to_register)
{
	auto status = nvrtcAddNameExpression(program_handle, global_to_register);
	throw_if_error(status, "Failed registering global entity " + ::std::string(global_to_register)
		+ " with " + identify(program_handle));
}

inline compilation_output_t compile(
	const char *                program_name,
	const const_cstrings_span & raw_options,
	handle_t                    program_handle)
{
	auto status = nvrtcCompileProgram(program_handle, (int) raw_options.size(), raw_options.data());
	bool succeeded = (status == status::named_t::success);
	if (not (succeeded or status == status::named_t::compilation_failure)) {
		throw rtc::runtime_error(status, "Failed invoking compiler for " + identify(program_handle));
	}
	constexpr bool do_own_handle{true};
	return compilation_output::detail_::wrap(program_handle, program_name, succeeded, do_own_handle);
}

// Note: The program_source _cannot_ be nullptr; if all of your source code is preincluded headers,
// pas the address of an empty string.
inline compilation_output_t compile(
	const char *program_name,
	const char *program_source,
	const_cstrings_span header_sources,
	const_cstrings_span header_names,
	const_cstrings_span raw_options,
	const_cstrings_span globals_to_register)
{
	assert(header_names.size() <= ::std::numeric_limits<int>::max());
	if (program_name == nullptr or *program_name == '\0') {
		throw ::std::invalid_argument("Attempt to compile a CUDA program without specifying a name");
	}
	// Note: Not rejecting empty/missing source, because we may be pre-including source files
	auto num_headers = (int) header_names.size();
	auto program_handle = create(
		program_name, program_source, num_headers, header_sources.data(), header_names.data());

	for (const auto global_to_register: globals_to_register) {
		register_global(program_handle, global_to_register);
	}

	// Note: compilation is outside of any context
	return compile(program_name, raw_options, program_handle);
}

} // namespace detail_

} // namespace program


/**
 * Wrapper class for a CUDA runtime-compilable program
 *
 * @note This class is a "reference type", not a "value type". Therefore, making changes
 * to the program is a const-respecting operation on this class.
 */
class program_t {

public: // getters

	const ::std::string& name() const { return name_; }
	const char* source() const { return source_; }
	const compilation_options_t& options() const { return options_; }
	// TODO: Think of a way to set compilation options without having
	// to break the statement, e.g. if options had a reflected enum value
	// or some such arrangement.
	compilation_options_t& options() { return options_; }
	const_cstrings_span header_names() const
	{
		return { headers_.names.data(), headers_.names.size()};
	}
	const_cstrings_span header_sources() const
	{
		return { headers_.sources.data(), headers_.sources.size()};
	}
	size_t num_headers() const { return headers_.sources.size(); }

public: // setters

	program_t& set_target(device::compute_capability_t target_compute_capability)
	{
		options_.set_target(target_compute_capability);
		return *this;
	}
	program_t& set_target(const device_t& device) { return set_target(device.compute_capability());}
	program_t& set_target(const context_t& context) { return set_target(context.device()); }
	program_t& clear_targets() { options_.targets_.clear(); return *this; }
	template <typename Container>
	program_t& set_targets(Container target_compute_capabilities)
	{
		clear_targets();
		for(const auto& compute_capability : target_compute_capabilities) {
			options_.add_target(compute_capability);
		}
		return *this;
	}
	program_t& add_target(device::compute_capability_t target_compute_capability)
	{
		options_.add_target(target_compute_capability);
		return *this;
	}
	void add_target(const device_t& device) { add_target(device.compute_capability()); }
	void add_target(const context_t& context) { add_target(context.device()); }

	program_t& set_source(const char* source) { source_ = source; return *this; }
	program_t& set_source(const ::std::string& source) { source_ = source.c_str(); return *this; }
	program_t& set_options(compilation_options_t options)
	{
		options_ = ::std::move(options);
		return *this;
	}

	template <typename HeaderNamesFwdIter, typename HeaderSourcesFwdIter>
	inline program_t& add_headers(
		HeaderNamesFwdIter header_names_start,
		HeaderNamesFwdIter header_names_end,
		HeaderSourcesFwdIter header_sources_start)
	{
		auto num_headers_to_add = header_names_end - header_names_start;
		auto new_num_headers = headers_.names.size() + num_headers_to_add;
#ifndef NDEBUG
		if (new_num_headers > ::std::numeric_limits<int>::max()) {
			throw ::std::invalid_argument("Cannot use more than "
										  + ::std::to_string(::std::numeric_limits<int>::max()) + " headers.");
		}
#endif
		headers_.names.reserve(new_num_headers);
		::std::copy_n(header_names_start, new_num_headers, ::std::back_inserter(headers_.names));
		headers_.sources.reserve(new_num_headers);
		::std::copy_n(header_sources_start, new_num_headers, ::std::back_inserter(headers_.sources));
		return *this;
	}

	template <typename HeaderNameAndSourceFwdIter>
	inline program_t& add_headers(
		HeaderNameAndSourceFwdIter named_headers_start,
		HeaderNameAndSourceFwdIter named_headers_end)
	{
		auto num_headers_to_add = named_headers_end - named_headers_start;
		auto new_num_headers = headers_.names.size() + num_headers_to_add;
#ifndef NDEBUG
		if (new_num_headers > ::std::numeric_limits<int>::max()) {
			throw ::std::invalid_argument("Cannot use more than "
										  + ::std::to_string(::std::numeric_limits<int>::max()) + " headers.");
		}
#endif
		headers_.names.reserve(new_num_headers);
		headers_.sources.reserve(new_num_headers);
		for(auto& pair_it = named_headers_start; pair_it < named_headers_end; pair_it++) {
			headers_.names.push_back(pair_it->first);
			headers_.sources.push_back(pair_it->second);
		}
		return *this;
	}

	const program_t& add_headers(
		const_cstrings_span header_names,
		const_cstrings_span header_sources)
	{
#ifndef NDEBUG
		if (header_names.size() != header_sources.size()) {
			throw ::std::invalid_argument(
				"Got " + ::std::to_string(header_names.size()) + " header names with "
				+ ::std::to_string(header_sources.size()));
		}
#endif
		return add_headers(header_names.cbegin(), header_names.cend(), header_sources.cbegin());
	}

	template <typename ContainerOfCStringPairs>
	program_t& add_headers(ContainerOfCStringPairs named_header_pairs)
	{
		return add_headers(::std::begin(named_header_pairs), ::std::end(named_header_pairs));
	}

	template <typename HeaderNamesFwdIter, typename HeaderSourcesFwdIter>
	program_t& set_headers(
		HeaderNamesFwdIter header_names_start,
		HeaderNamesFwdIter header_names_end,
		HeaderSourcesFwdIter header_sources_start)
	{
		clear_headers();
		return add_headers(header_names_start, header_names_end, header_sources_start);
	}

	program_t& set_headers(
		const_cstrings_span header_names,
		const_cstrings_span header_sources)
	{
#ifndef NDEBUG
		if (header_names.size() != header_sources.size()) {
			throw ::std::invalid_argument(
				"Got " + ::std::to_string(header_names.size()) + " header names with "
				+ ::std::to_string(header_sources.size()));
		}
#endif
		return set_headers(header_names.cbegin(), header_names.cend(), header_sources.cbegin());
	}

	template <typename ContainerOfCStringPairs>
	program_t& set_headers(ContainerOfCStringPairs named_header_pairs)
	{
		clear_headers();
		return add_headers(named_header_pairs);
	}


	program_t& clear_headers()
	{
		headers_.names.clear();
		headers_.sources.clear();
		return *this;
	}

	program_t& clear_options() { options_ = {}; return *this; }

public:

	// TODO: Support specifying all compilation option in a single string and parsing it

	compilation_output_t compile() const
	{
		if ((source_ == nullptr or *source_ == '\0') and options_.preinclude_files.empty()) {
			throw ::std::invalid_argument("Attempt to compile a CUDA program without any source code");
		}
		auto marshalled_options = marshal(options_);
		::std::vector<const char*> option_ptrs = marshalled_options.option_ptrs();
		return program::detail_::compile(
			name_.c_str(),
			source_ == nullptr ? "" : source_,
			{headers_.sources.data(), headers_.sources.size()},
			{headers_.names.data(), headers_.names.size()},
			{option_ptrs.data(), option_ptrs.size()},
			{globals_to_register_.data(), globals_to_register_.size()});
	}

	/**
	 * @brief Register a pre-mangled name of a kernel, to make available for use
	 * after compilation
	 *
	 * @param name The text of an expression, e.g. "my_global_func()", "f1", "N1::N2::n2",
	 *
	 */
	program_t& add_registered_global(const char* unmangled_name)
	{
		globals_to_register_.push_back(unmangled_name);
		return *this;
	}
	program_t& add_registered_global(const ::std::string& unmangled_name)
	{
		globals_to_register_.push_back(unmangled_name.c_str());
		return *this;
	}

	template <typename Container>
	program_t& add_registered_globals(Container&& globals_to_register)
	{
		globals_to_register_.reserve(globals_to_register_.size() + globals_to_register.size());
		::std::copy(globals_to_register.cbegin(), globals_to_register.cend(), ::std::back_inserter(globals_to_register_));
		return *this;
	}

public: // constructors and destructor
	program_t(::std::string name) : name_(::std::move(name)) {};
	program_t(const program_t&) = default;
	program_t(program_t&&) = default;
	~program_t() = default;

public: // operators

	program_t& operator=(const program_t& other) = default;
	program_t& operator=(program_t&& other) = default;

protected: // data members
	const char*           source_ { nullptr };
	::std::string         name_;
	compilation_options_t options_;
	struct {
		::std::vector<const char*> names;
		::std::vector<const char*> sources;
	} headers_;
	::std::vector<const char*> globals_to_register_;
}; // class program_t

namespace program {

inline program_t create(const char* program_name)
{
	return program_t(program_name);
}

inline program_t create(const ::std::string program_name)
{
	return program_t(program_name);
}

} // namespace program

#if CUDA_VERSION >= 11020
inline dynarray<device::compute_capability_t>
supported_targets()
{
	int num_supported_archs;
	auto status = nvrtcGetNumSupportedArchs(&num_supported_archs);
	throw_if_error(status, "Failed obtaining the number of target NVRTC architectures");
	auto raw_archs = ::std::unique_ptr<int[]>(new int[num_supported_archs]);
	status = nvrtcGetSupportedArchs(raw_archs.get());
	throw_if_error(status, "Failed obtaining the architectures supported by NVRTC");
	dynarray<device::compute_capability_t> result;
	result.reserve(num_supported_archs);
	::std::transform(raw_archs.get(), raw_archs.get() + num_supported_archs, ::std::back_inserter(result),
		[](int raw_arch) { return device::compute_capability_t::from_combined_number(raw_arch); });
	return result;
}
#endif

} // namespace rtc

} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVRTC_PROGRAM_HPP_
