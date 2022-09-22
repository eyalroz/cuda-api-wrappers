/**
 * @file
 *
 * @brief Contains the @ref cuda::rtc::program_t class and related code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NVRTC_PROGRAM_HPP_
#define CUDA_API_WRAPPERS_NVRTC_PROGRAM_HPP_

#include <cuda/rtc/compilation_options.hpp>
#include <cuda/rtc/compilation_output.hpp>
#include <cuda/rtc/error.hpp>
#include <cuda/rtc/types.hpp>
#include <cuda/api.hpp>

#include <vector>

namespace cuda {

namespace rtc {

class ptx_program_t;


namespace detail_ {

::std::string identify(const ptx_program_t &program);

} // namespace detail_

namespace program {

namespace detail_ {

inline program::handle_t<cuda_cpp> create_cuda_cpp(
	const char *program_name,
	const char *program_source,
	int num_headers,
	const char *const *header_sources,
	const char *const *header_names)
{
	program::handle_t<cuda_cpp> program_handle;
	auto status = nvrtcCreateProgram(&program_handle, program_source, program_name, num_headers, header_sources, header_names);
	throw_if_rtc_error_lazy(cuda_cpp, status, "Failed creating " + detail_::identify<cuda_cpp>(program_name));
	return program_handle;
}

inline program::handle_t<source_kind_t::ptx> create_ptx(
	const char *program_name,
	string_view program_source)
{
	program::handle_t<source_kind_t::ptx> program_handle;
	auto status = nvPTXCompilerCreate(&program_handle, program_source.size(), program_source.data());
	throw_if_rtc_error_lazy(source_kind_t::ptx, status, "Failed creating " + detail_::identify<source_kind_t::ptx>(program_name));
	return program_handle;
}

template <source_kind_t Kind>
inline program::handle_t<Kind> create(
	const char *program_name,
	string_view program_source,
	int num_headers = 0,
	const char *const *header_sources = nullptr,
	const char *const *header_names = nullptr)
{
	return (Kind == cuda_cpp) ?
		(program::handle_t<Kind>) create_cuda_cpp(program_name, program_source.data(), num_headers, header_sources, header_names) :
		(program::handle_t<Kind>) create_ptx(program_name, program_source);
}

inline void register_global(handle_t<cuda_cpp> program_handle, const char *global_to_register)
{
	auto status = nvrtcAddNameExpression(program_handle, global_to_register);
	throw_if_rtc_error_lazy(cuda_cpp, status, "Failed registering global entity " + ::std::string(global_to_register)
		+ " with " + identify<cuda_cpp>(program_handle));
}

template <source_kind_t Kind>
inline compilation_output_t<Kind> compile(
	const char *                program_name,
	const const_cstrings_span&  raw_options,
	handle_t<Kind>              program_handle)
{
	status_t<Kind> status = (Kind == cuda_cpp) ?
		(status_t<Kind>) nvrtcCompileProgram((handle_t<cuda_cpp>)program_handle, (int) raw_options.size(), raw_options.data()) :
		(status_t<Kind>) nvPTXCompilerCompile((handle_t<source_kind_t::ptx>)program_handle, (int) raw_options.size(), raw_options.data());
	bool succeeded = is_success<Kind>(status);
	if (not (succeeded or (status == (status_t<Kind>) status::named_t<Kind>::compilation_failure))) {
		throw rtc::runtime_error<Kind>(status, "Failed invoking compiler for " + identify<Kind>(program_handle));
	}
	constexpr bool do_own_handle{true};
	return compilation_output::detail_::wrap<Kind>(program_handle, program_name, succeeded, do_own_handle);
}

// Note: The program_source _cannot_ be nullptr; if all of your source code is preincluded headers,
// pas the address of an empty string.
inline compilation_output_t<cuda_cpp> compile(
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
	auto program_handle = create_cuda_cpp(
		program_name, program_source, num_headers, header_sources.data(), header_names.data());

	for (const auto global_to_register: globals_to_register) {
		register_global(program_handle, global_to_register);
	}

	// Note: compilation is outside of any context
	return compile<cuda_cpp>(program_name, raw_options, program_handle);
}

inline compilation_output_t<source_kind_t::ptx> compile_ptx(
	const char *program_name,
	const char *program_source,
	const_cstrings_span raw_options)
{
	if (program_name == nullptr or *program_name == '\0') {
		throw ::std::invalid_argument("Attempt to compile a CUDA program without specifying a name");
	}
	// Note: Not rejecting empty/missing source, because we may be pre-including source files
	auto program_handle = create_ptx(program_name, program_source);

	// Note: compilation is outside of any context
	return compile<source_kind_t::ptx>(program_name, raw_options, program_handle);
}

} // namespace detail_

} // namespace program

template <source_kind_t Kind>
class program_base_t {
public: // types and constants
	constexpr static const source_kind_t source_kind { Kind };
	using handle_type = program::handle_t<source_kind>;
	using status_type = status_t<source_kind>;

public: // getters

	const ::std::string& name() const { return name_; }
	const char* source() const { return source_; }
	const compilation_options_t<Kind>& options() const { return options_; }
	// TODO: Think of a way to set compilation options without having
	// to break the statement, e.g. if options had a reflected enum value
	// or some such arrangement.
	compilation_options_t<Kind>& options() { return options_; }

public: // constructors and destructor
	program_base_t(::std::string name) : name_(::std::move(name)) {};
	program_base_t(const program_base_t&) noexcept = default;
	program_base_t(program_base_t&&) noexcept = default;
	~program_base_t() = default;

public: // operators

	program_base_t& operator=(const program_base_t& other) noexcept = default;
	program_base_t& operator=(program_base_t&& other) noexcept = default;

protected: // data members
	const char*           source_ { nullptr };
	::std::string         name_;
	compilation_options_t<Kind> options_;
}; // program_base_t

template <source_kind_t Kind>
class program_t;

/**
 * Wrapper class for a CUDA runtime-compilable program
 *
 * @note This class is a "reference type", not a "value type". Therefore, making changes
 * to the program is a const-respecting operation on this class.
 *
 */
template <>
class program_t<cuda_cpp> : public program_base_t<cuda_cpp> {
public: // types
	using parent = program_base_t<source_kind>;

public: // getters

	const_cstrings_span header_names() const
	{
		return { headers_.names.data(),headers_.names.size()};
	}
	const_cstrings_span header_sources() const
	{
		return { headers_.sources.data(), headers_.sources.size()};
	}
	size_t num_headers() const { return headers_.sources.size(); }

public: // setters - duplicated with PTX programs

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
	program_t& set_options(compilation_options_t<source_kind> options)
	{
		options_ = ::std::move(options);
		return *this;
	}

protected:
	template <typename String>
	static inline void check_string_type()
	{
		using no_cref_string_type = typename ::std::remove_const<typename ::std::remove_reference<String>::type>::type;
		static_assert(
			::std::is_same<no_cref_string_type, const char*>::value or
			::std::is_same<no_cref_string_type, char*>::value or
			::std::is_same<String, const ::std::string&>::value or
			::std::is_same<String, ::std::string&>::value,
			"Cannot use this type for a named header name or source; use char*, const char* or a "
			"reference to a string you own"
		);
	}

	// Note: All methods involved in adding headers - which eventually call one of the
	// three adders of each kind here - are written carefully to support both C-style strings
	// and lvalue references to ::std::string's - but _not_ rvalue strings or rvalue string
	// references, as the latter are not owned by the caller, and this class' code does not
	// make a copy or take ownership. If you make any changes, you must be very careful not
	// to _copy_ anything by mistake, but rather carry forward reference-types all the way
	// to here.

	void add_header_name_  (const char* name)            { headers_.names.emplace_back(name); }
	void add_header_name_  (const ::std::string& name)   { add_header_name_(name.c_str()); }
	void add_header_name_  (::std::string&& name) = delete;

	void add_header_source_(const char* source)          { headers_.sources.emplace_back(source); }
	void add_header_source_(const ::std::string& source) { add_header_source_(source.c_str()); }
	void add_header_source_(::std::string&& source) = delete;

public: // mutators
	template <typename String1, typename String2>
	program_t& add_header(String1&& name, String2&& source)
	{
		add_header_name_(name);
		add_header_source_(source);
		return *this;
	}

	template <typename String1, typename String2>
	program_t& add_header(const ::std::pair<String1, String2>& name_and_source)
	{
		add_header_name_(name_and_source.first);
		add_header_source_(name_and_source.second);
		return *this;
	}

	template <typename String1, typename String2>
	program_t& add_header(::std::pair<String1, String2>&& name_and_source)
	{
		check_string_type<String1>();
		check_string_type<String2>();
		return add_header(name_and_source);
	}

	template <typename RangeOfNames, typename RangeOfSources>
	const program_t& add_headers(
		RangeOfNames   header_names,
		RangeOfSources header_sources)
	{
		check_string_type<typename RangeOfNames::const_reference>();
		check_string_type<typename RangeOfSources::const_reference>();
#ifndef NDEBUG
		if (header_names.size() != header_sources.size()) {
			throw ::std::invalid_argument(
				"Got a different number of header names (" + ::std::to_string(header_names.size())
				+ ") and header source (" + ::std::to_string(header_sources.size()) + ')');
		}
#endif
		auto new_num_headers = headers_.names.size() + header_names.size();
#ifndef NDEBUG
		if (new_num_headers > ::std::numeric_limits<int>::max()) {
			throw ::std::invalid_argument("Cannot use more than "
										  + ::std::to_string(::std::numeric_limits<int>::max()) + " headers.");
		}
#endif
		headers_.names.reserve(new_num_headers);
		headers_.sources.reserve(new_num_headers);
		// TODO: Use a zip iterator
		for(auto name_it = header_names.cbegin(), source_it = header_sources.cbegin();
		    name_it < header_names.cend();
			name_it++, source_it++) {
			add_header(*name_it, *source_it);
		}
		return *this;
	}

	template <typename RangeOfNameAndSourcePairs>
	program_t& add_headers(RangeOfNameAndSourcePairs&& named_header_pairs)
	{
		// TODO: Accept ranges without a size method and no iterator arithmetic
		auto num_headers_to_add = named_header_pairs.size();
		auto new_num_headers = headers_.names.size() + num_headers_to_add;
#ifndef NDEBUG
		if (new_num_headers > ::std::numeric_limits<int>::max()) {
			throw ::std::invalid_argument("Cannot use more than "
										  + ::std::to_string(::std::numeric_limits<int>::max()) + " headers.");
		}
#endif
		headers_.names.reserve(new_num_headers);
		headers_.sources.reserve(new_num_headers);
		// Using auto&& to notice the case of getting rvalue references  (which we would like to reject)
		for(auto&& pair : named_header_pairs) {
			add_header(pair.first, pair.second);
		}
		return *this;
	}

	template <typename RangeOfNames, typename RangeOfSources>
	const program_t& set_headers(
		RangeOfNames&&   names,
		RangeOfSources&& sources)
	{
		clear_headers();
		return add_headers(names, sources);
	}

	template <typename RangeOfNameAndSourcePairs>
	program_t& set_headers(RangeOfNameAndSourcePairs&& named_header_pairs)
	{
		clear_headers();
		add_headers(named_header_pairs);
		return *this;
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

	compilation_output_t<cuda_cpp> compile() const
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
	program_t(::std::string name) : program_base_t(std::move(name)) {}
	program_t(const program_t&) = default;
	program_t(program_t&&) = default;
	~program_t() = default;

public: // operators

	program_t& operator=(const program_t& other) = default;
	program_t& operator=(program_t&& other) = default;

protected: // data members
	struct {
		::std::vector<const char*> names;
		::std::vector<const char*> sources;
	} headers_;
	::std::vector<const char*> globals_to_register_;
}; // class program_t<cuda_cpp>

#if CUDA_VERSION >= 11010

template <>
class program_t<source_kind_t::ptx> : public program_base_t<source_kind_t::ptx> {
public: // types
	using parent = program_base_t<source_kind>;

public: // setters - duplicated with CUDA-C++/NVRTC programs

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
	program_t& set_options(compilation_options_t<source_kind> options)
	{
		options_ = ::std::move(options);
		return *this;
	}
	program_t& clear_options() { options_ = {}; return *this; }

public:

	// TODO: Support specifying all compilation option in a single string and parsing it

	compilation_output_t<source_kind_t::ptx> compile() const
	{
		if (source_ == nullptr or *source_ == '\0') {
			throw ::std::invalid_argument("Attempt to compile a CUDA program without any source code");
		}
		auto marshalled_options = marshal(options_);
		::std::vector<const char*> option_ptrs = marshalled_options.option_ptrs();
		return program::detail_::compile_ptx(
			name_.c_str(),
			source_,
			{option_ptrs.data(), option_ptrs.size()});
	}


public: // constructors and destructor
	program_t(::std::string name) : program_base_t(std::move(name)) {}
	program_t(const program_t&) = default;
	program_t(program_t&&) = default;
	~program_t() = default;

public: // operators

	program_t& operator=(const program_t& other) = default;
	program_t& operator=(program_t&& other) = default;
}; // class program_t<source_kind_t::ptx>

#endif // CUDA_VERSION >= 11010

namespace program {

template <source_kind_t Kind>
inline program_t<Kind> create(const char* program_name)
{
	return program_t<Kind>(program_name);
}

template <source_kind_t Kind>
inline program_t<Kind> create(const ::std::string& program_name)
{
	return program_t<Kind>(program_name);
}

} // namespace program

#if CUDA_VERSION >= 11020
inline dynarray<device::compute_capability_t>
supported_targets()
{
	int num_supported_archs;
	auto status = nvrtcGetNumSupportedArchs(&num_supported_archs);
	throw_if_error<cuda_cpp>(status, "Failed obtaining the number of target NVRTC architectures");
	auto raw_archs = ::std::unique_ptr<int[]>(new int[num_supported_archs]);
	status = nvrtcGetSupportedArchs(raw_archs.get());
	throw_if_error<cuda_cpp>(status, "Failed obtaining the architectures supported by NVRTC");
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
