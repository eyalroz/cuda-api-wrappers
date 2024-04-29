/**
 * @file
 *
 * @brief Contains the @ref cuda::rtc::program_t class and related code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_RTC_PROGRAM_HPP_
#define CUDA_API_WRAPPERS_RTC_PROGRAM_HPP_

#include "compilation_options.hpp"
#include "compilation_output.hpp"
#include "error.hpp"
#include "types.hpp"
#include "../api.hpp"

#include <vector>

namespace cuda {

namespace rtc {

namespace program {

namespace detail_ {

/**
 * Create a new program object from source code
 *
 * @tparam Kind We can create a program with any one of the (two...) kinds of supported
 *     source code
 * @param program_name arbitrary identifier to recognize the program by; it's suggested
 *     not to get too crazy
 * @param program_source The source code of the program, possibly with include directives
 *     in the case of C++
 * @param num_headers The number of pairs of header "file" names and header content strings
 * @param header_sources Pointers to nul-terminated per-header source code
 * @param header_names Pointers to nul-terminated names of the different headers
 */
template <source_kind_t Kind = cuda_cpp>
inline program::handle_t<Kind> create(
	const char *program_name,
	string_view program_source,
	int num_headers = 0,
	const char *const *header_sources = nullptr,
	const char *const *header_names = nullptr);

template <> inline program::handle_t<cuda_cpp> create<cuda_cpp>(
	const char *program_name, string_view program_source, int num_headers,
	const char *const *header_sources, const char *const *header_names)
{
	program::handle_t<cuda_cpp> program_handle;
	auto status = nvrtcCreateProgram(&program_handle, program_source.data(), program_name, num_headers, header_sources, header_names);
	throw_if_rtc_error_lazy(cuda_cpp, status, "Failed creating " + detail_::identify<cuda_cpp>(program_name));
	return program_handle;
}

#if CUDA_VERSION >= 11010
template <> inline program::handle_t<ptx> create<ptx>(
	const char *program_name, string_view program_source,
	int, const char *const *, const char *const *)
{
	program::handle_t<ptx> program_handle;
	auto status = nvPTXCompilerCreate(&program_handle, program_source.size(), program_source.data());
	throw_if_rtc_error_lazy(ptx, status, "Failed creating " + detail_::identify<ptx>(program_name));
	return program_handle;
}
#endif // CUDA_VERSION >= 11010

/// Have NVRTC add the specified global to those accessible/usable after compilation
inline void register_global(handle_t<cuda_cpp> program_handle, const char *global_to_register)
{
	auto status = nvrtcAddNameExpression(program_handle, global_to_register);
	throw_if_rtc_error_lazy(cuda_cpp, status, "Failed registering global entity " + ::std::string(global_to_register)
		+ " with " + identify<cuda_cpp>(program_handle));
}

/// Splice multiple raw string options together with a ' ' separator character, and
/// surrounding each option with double-quotes
inline ::std::string get_concatenated_options(const const_cstrings_span& raw_options)
{
	static ::std::ostringstream oss;
	oss.str("");
	for (const auto option: raw_options) {
		oss << " \"" << option << '\"';
	}
	return oss.str();
}

template <source_kind_t Kind>
inline void maybe_handle_invalid_option(
	status_t<Kind>,
	const char *,
	const const_cstrings_span&,
	handle_t<Kind>)
{ }

template <>
inline void maybe_handle_invalid_option<cuda_cpp>(
	status_t<cuda_cpp>          status,
	const char *                program_name,
	const const_cstrings_span&  raw_options,
	handle_t<cuda_cpp>          program_handle)
{
	if (status == static_cast<status_t<cuda_cpp>>(status::named_t<cuda_cpp>::invalid_option)) {
		throw rtc::runtime_error<cuda_cpp>::with_message_override(status,
			"Compilation options rejected when compiling " + identify<cuda_cpp>(program_handle, program_name) + ':'
			+ get_concatenated_options(raw_options));
	}
}

template <source_kind_t Kind>
inline status_t<Kind> compile_and_return_status(
	handle_t<Kind> program_handle,
	const const_cstrings_span& raw_options);

#if CUDA_VERSION >= 11010
template <>
inline status_t<ptx> compile_and_return_status<ptx>(
	handle_t<ptx> program_handle,
	const const_cstrings_span& raw_options)
{
	return nvPTXCompilerCompile(program_handle, static_cast<int>(raw_options.size()), raw_options.data());
}
#endif

template <>
inline status_t<cuda_cpp> compile_and_return_status<cuda_cpp>(
	handle_t<cuda_cpp> program_handle,
	const const_cstrings_span& raw_options)
{
	return nvrtcCompileProgram(program_handle, static_cast<int>(raw_options.size()), raw_options.data());
}


template <source_kind_t Kind>
inline compilation_output_t<Kind> compile(
	const char *                program_name,
	const const_cstrings_span&  raw_options,
	handle_t<Kind>              program_handle)
{
	auto status = compile_and_return_status<Kind>(program_handle, raw_options);
	bool succeeded = is_success<Kind>(status);
	switch(status) {
	case status::named_t<Kind>::success:
	case status::named_t<Kind>::compilation_failure:
		return compilation_output::detail_::wrap<Kind>(program_handle, program_name, succeeded, do_take_ownership);
	default:
		maybe_handle_invalid_option<Kind>(status, program_name, raw_options, program_handle);
		throw rtc::runtime_error<Kind>(status, "Failed invoking compiler for " + identify<Kind>(program_handle, program_name));
	}
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
	auto num_headers = static_cast<int>(header_names.size());
	auto program_handle = create<cuda_cpp>(
		program_name, program_source, num_headers, header_sources.data(), header_names.data());

	for (const auto global_to_register: globals_to_register) {
		register_global(program_handle, global_to_register);
	}

	// Note: compilation is outside of any context
	return compile<cuda_cpp>(program_name, raw_options, program_handle);
}

#if CUDA_VERSION >= 11010
inline compilation_output_t<ptx> compile_ptx(
	const char *program_name,
	const char *program_source,
	const_cstrings_span raw_options)
{
	if (program_name == nullptr or *program_name == '\0') {
		throw ::std::invalid_argument("Attempt to compile a CUDA program without specifying a name");
	}
	// Note: Not rejecting empty/missing source, because we may be pre-including source files
	auto program_handle = create<ptx>(program_name, program_source);

	// Note: compilation is outside of any context
	return compile<ptx>(program_name, raw_options, program_handle);
}
#endif // CUDA_VERSION >= 11010

template <source_kind_t Kind>
class base_t {
public: // types and constants
	constexpr static const source_kind_t source_kind { Kind };
	using handle_type = program::handle_t<source_kind>;
	using status_type = status_t<source_kind>;

public: // getters
    /// Getters for the constituent object fields
    ///@{
	const ::std::string& name() const { return name_; }

	/// Full source code of the program (possibly with preprocessor directives such as `#include`)
	const char* source() const { return source_; }

	/// Compilation options to be passed to the JIT compiling library along with the source code
	const compilation_options_t<Kind>& options() const { return options_; }
	// TODO: Think of a way to set compilation options without having
	// to break the statement, e.g. if options had a reflected enum value
	// or some such arrangement.

	/// Compilation options to be passed to the JIT compiling library along with the source code
	compilation_options_t<Kind>& options() { return options_; }
	///@}

public: // constructors and destructor
	explicit base_t(::std::string name) : name_(::std::move(name)) {};
	base_t(const base_t&) = default;
	base_t(base_t&&) noexcept = default;
	~base_t() = default;

public: // operators

	base_t& operator=(const base_t& other) noexcept = default;
	base_t& operator=(base_t&& other) noexcept = default;

protected: // data members
	const char*           source_ { nullptr };
	::std::string         name_;
	compilation_options_t<Kind> options_;
}; // base_t

} // namespace detail_

} // namespace program

template <source_kind_t Kind>
class program_t;

/**
 * Wrapper class for a CUDA runtime-compilable program
 *
 * @note This class is a "reference type", not a "value type". Therefore, making changes
 * to the program is a const-respecting operation on this class.
 *
 * @note Many of this class' methods could have been placed in the base class, and are
 * "duplicated" in program_t<ptx> - except that they return the program object itself,
 * allowing for builder-pattern-like use.
 */
template <>
class program_t<cuda_cpp> : public program::detail_::base_t<cuda_cpp> {
public: // types
	using parent = base_t<source_kind>;

public: // getters

	/// Names of the "memoized"/off-file-system headers made available to the program
	/// (and usable as identifiers for `#include` directives)
	const_cstrings_span header_names() const
	{
		return { headers_.names.data(),headers_.names.size()};
	}

	/// Sources of the "memoized"/off-file-system headers made available to the program
	/// (and usable as identifiers for `#include` directives)
	///
	/// @note each header source string corresponds to the name of the same index 
	/// accessible via {@ref header_names()}.
	const_cstrings_span header_sources() const
	{
		return { headers_.sources.data(), headers_.sources.size()};
	}

	/// @returns the number of memoized/off-the-file-system headers made available
	/// to the program
	size_t num_headers() const { return headers_.sources.size(); }

public: // setters - duplicated with PTX programs

	/// Have the compilation produce code for devices with a given compute capability
	program_t& set_target(device::compute_capability_t target_compute_capability)
	{
		options_.set_target(target_compute_capability);
		return *this;
	}

	/// Have the compilation produce code for devices with the same compute capability
	/// as a given device
	program_t& set_target(const device_t& device) { return set_target(device.compute_capability());}

	/// Have the compilation produce code for devices with the same compute capability
	/// as the device of a given context
	program_t& set_target(const context_t& context) { return set_target(context.device()); }

	/// Remove all compute capabilities which were chosen to have code produced for them
	/// by the compilation
	program_t& clear_targets() { options_.targets_.clear(); return *this; }

	/// Remove all compute capabilities which were chosen to have code produced for them
	/// by the compilation
	template <typename Container>
	program_t& set_targets(Container target_compute_capabilities)
	{
		clear_targets();
		for(const auto& compute_capability : target_compute_capabilities) {
			options_.add_target(compute_capability);
		}
		return *this;
	}

	/// Have the compilation also produce code for devices with a given compute
	/// capability
	program_t& add_target(device::compute_capability_t target_compute_capability)
	{
		options_.add_target(target_compute_capability);
		return *this;
	}

	/// Have the compilation also produce code for devices with the same compute
	/// capability as a given device
	void add_target(const device_t& device) { add_target(device.compute_capability()); }

	/// Have the compilation also produce code for devices with the same compute
	/// capability as the device of a given context
	void add_target(const context_t& context) { add_target(context.device()); }

	program_t& set_source(const char* source) { source_ = source; return *this; }
	program_t& set_source(const ::std::string& source) { source_ = source.c_str(); return *this; }
	program_t& set_options(const compilation_options_t<source_kind>& options)
	{
		options_ = options;
		return *this;
	}
	program_t& set_options(compilation_options_t<source_kind>&& options)
	{
		options_ = ::std::move(options);
		return *this;
	}

protected:
	template <typename String>
	static void check_string_type()
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
	/**
	 * Adds another "memoized" header to the program
	 *
	 * @param name The header name for use in `#include` directives
	 * @param source The full source code of the header "file", possibly with its own
	 *     preprocessor directives (e.g. `#include`).
	 *
	 * @note "names" with path separators can be used, but are discouraged
	 */
	template <typename String1, typename String2>
	program_t& add_header(String1&& name, String2&& source)
	{
		add_header_name_(name);
		add_header_source_(source);
		return *this;
	}

	/**
	 * Adds another "memoized" header to the program
	 *
	 * @param name_and_source A pair of strings, one being the name for use in `#include`
	 *     directives, the other being the full source code of the header "file", possibly
	 *     with its own preprocessor directives (e.g. `#include`).
	 *
	 * @note "names" with path separators can be used, but are discouraged
	 */
	template <typename String1, typename String2>
	program_t& add_header(const ::std::pair<String1, String2>& name_and_source)
	{
		add_header_name_(name_and_source.first);
		add_header_source_(name_and_source.second);
		return *this;
	}

	/// @copydoc add_header<String1, String2>(String1&&, String2&&)
	template <typename String1, typename String2>
	program_t& add_header(::std::pair<String1, String2>&& name_and_source)
	{
		check_string_type<String1>();
		check_string_type<String2>();
		return add_header(name_and_source);
	}

	/**
	 * Adds multiple "memoized" headers to the program
	 *
	 * @param name Names of the headers, for use in `#include` directives
	 * @param source The full source code of each of the header "file", possibly
	 *     with their own preprocessor directivess.
	 *
	 * @note "names" with path separators can be used, but are discouraged
	 */
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
		for(auto name_it = header_names.begin(), source_it = header_sources.begin();
		    name_it < header_names.end();
			name_it++, source_it++) {
			add_header(*name_it, *source_it);
		}
		return *this;
	}

	/**
	 * Adds multiple "memoized" headers to the program
	 *
	 * @param name_and_source_pairs A container of pairs of strings, each being made
	 *     up of a name for use in `#include` directives, and the full source code
	 *     of the header "file", possibly with its own preprocessor directives.
	 *
	 * @note "names" with path separators can be used, but are discouraged
	 */
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

	/**
	 * Replaces the set of "memoized" headers used in the program's compilation
	 *
	 * @param name Names of the headers, for use in `#include` directives
	 * @param source The full source code of each of the header "file", possibly
	 *     with their own preprocessor directivess.
	 *
	 * @note "names" with path separators can be used, but are discouraged
	 */
	template <typename RangeOfNames, typename RangeOfSources>
	const program_t& set_headers(
		RangeOfNames&&   names,
		RangeOfSources&& sources)
	{
		clear_headers();
		return add_headers(names, sources);
	}

	/**
	 * Replaces the set of "memoized" headers used in the program's compilation
	 *
	 * @param name_and_source_pairs A container of pairs of strings, each being made
	 *     up of a name for use in `#include` directives, and the full source code
	 *     of the header "file", possibly with its own preprocessor directives.
	 *
	 * @note "names" with path separators can be used, but are discouraged
	 */
	template <typename RangeOfNameAndSourcePairs>
	program_t& set_headers(RangeOfNameAndSourcePairs&& named_header_pairs)
	{
		clear_headers();
		add_headers(named_header_pairs);
		return *this;
	}

	/// Removes all "memoized" headers to be used in the program's compilation
	program_t& clear_headers()
	{
		headers_.names.clear();
		headers_.sources.clear();
		return *this;
	}

	/// Clears any forced values of compilation options, reverting the compilation
	/// to the default values
	program_t& clear_options() { options_ = {}; return *this; }

public:

	// TODO: Support specifying all compilation option in a single string and parsing it

	/**
	 * Compiles the program represented by this object (which, until this point, is
	 * just a bunch of unrelated sources and options).
	 *
	 * @note Carefully examines the @ref compilation_output_t class to understand what
	 * exactly the compilation produces.
	 */
	compilation_output_t<cuda_cpp> compile() const
	{
		if ((source_ == nullptr or *source_ == '\0') and options_.preinclude_files.empty()) {
			throw ::std::invalid_argument("Attempt to compile a CUDA program without any source code");
		}
		auto marshalled_options = detail_::marshal(options_);
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
	 * @brief Register a pre-mangled name of a global, to make available for use
	 * after compilation
	 *
	 * @param unmangled_name The text of an expression, e.g. "my_global_func()", "f1", "N1::N2::n2",
	 *
	 * @note The name must continue to exist past the compilation of the program - as it is not copied,
	 * only referenced
	 */
	program_t& add_registered_global(const char* unmangled_name)
	{
		globals_to_register_.push_back(unmangled_name);
		return *this;
	}

	/// @copydoc add_registered_global(const char*)
	program_t& add_registered_global(const ::std::string& unmangled_name)
	{
		globals_to_register_.push_back(unmangled_name.c_str());
		return *this;
	}
	// TODO: Accept string_view's with C++17

	/**
	 * @brief Register multiple pre-mangled names of global, to make available for use
	 * after compilation
	 *
	 * @param globals_to_register a container of elements constituting the text of an expression
	 * identifying a global, e.g. "my_global_func()", "f1", "N1::N2::n2",
	 *
	 * @note All names in the container must continue to exist past the compilation of the
	 * program - as they are not copied, only referenced. Thus, as a safety precaution, we
	 * also assume the container continues to exist
	 */
	template <typename Container>
	program_t& add_registered_globals(const Container& globals_to_register)
	{
		globals_to_register_.reserve(globals_to_register_.size() + globals_to_register.size());
		for(const auto& global_name : globals_to_register) {
			add_registered_global(global_name);
		}
		return *this;
	}

	/// @copydic add_registered_globals(const Container&)
	template <typename Container>
	program_t& add_registered_globals(Container&& globals_to_register)
	{
		static_assert(::std::is_same<typename Container::value_type, const char*>::value,
			"For an rvalue container, we only accept raw C strings as the value type, to prevent"
			"the possible passing of string-like objects at the end of their lifetime");
		return add_registered_globals(static_cast<const Container&>(globals_to_register));
	}

public: // constructors and destructor
	program_t(::std::string name) : base_t(::std::move(name)) {}
	program_t(const program_t&) = default;
	program_t(program_t&&) = default;
	~program_t() = default;

public: // operators
	///@cond
	program_t& operator=(const program_t& other) = default;
	program_t& operator=(program_t&& other) = default;
	///@endcond

protected: // data members
	struct {
		::std::vector<const char*> names;
		::std::vector<const char*> sources;
	} headers_;
	::std::vector<const char*> globals_to_register_;
}; // class program_t<cuda_cpp>

#if CUDA_VERSION >= 11010

/**
 * Wrapper class for a CUDA PTX (runtime-compilable) program
 *
 * @note This class is a "reference type", not a "value type". Therefore, making changes
 * to the program is a const-respecting operation on this class.
 *
 * @note Many of this class' methods could have been placed in the base class, and are
 * "duplicated" in program_t<ptx> - except that they return the program object itself,
 * allowing for builder-pattern-like use.
 */
template <>
class program_t<ptx> : public program::detail_::base_t<ptx> {
public: // types
	///@cond
	using parent = program::detail_::base_t<source_kind>;
	///@endcond

public: // setters - duplicated with CUDA-C++/NVRTC programs

	/// @copydoc program_t<cuda_cpp>::set_target(device::compute_capability_t)
	program_t& set_target(device::compute_capability_t target_compute_capability)
	{
		options_.set_target(target_compute_capability);
		return *this;
	}

	/// @copydoc program_t<cuda_cpp>::set_target(const device_t&)
	program_t& set_target(const device_t& device) { return set_target(device.compute_capability());}

	/// @copydoc program_t<cuda_cpp>::set_target(const context_t&)
	program_t& set_target(const context_t& context) { return set_target(context.device()); }

	/// @copydoc program_t<cuda_cpp>::clear_targets()
	program_t& clear_targets() { options_.targets_.clear(); return *this; }

	/// @copydoc program_t<cuda_cpp>::set_targets<Container>(Container)
	template <typename Container>
	program_t& set_targets(Container target_compute_capabilities)
	{
		clear_targets();
		for(const auto& compute_capability : target_compute_capabilities) {
			options_.add_target(compute_capability);
		}
		return *this;
	}

	/// @copydoc program_t<cuda_cpp>::add_target(device::compute_capability_t)
	program_t& add_target(device::compute_capability_t target_compute_capability)
	{
		options_.add_target(target_compute_capability);
		return *this;
	}

	/// @copydoc program_t<cuda_cpp>::clear_targets()
	void add_target(const device_t& device) { add_target(device.compute_capability()); }

	/// @copydoc program_t<cuda_cpp>::set_targets<Container>(Container)
	void add_target(const context_t& context) { add_target(context.device()); }

	/// @copydoc program_t<cuda_cpp>::set_source(char const*)
	program_t& set_source(char const* source) { source_ = source; return *this; }

	/// @copydoc program_t<cuda_cpp>::set_source(const ::std::string&)
	program_t& set_source(const ::std::string& source) { source_ = source.c_str(); return *this; }

	/// @copydoc program_t<cuda_cpp>::set_options(compilation_options_t<ptx>)
	program_t& set_options(compilation_options_t<source_kind> options)
	{
		options_ = ::std::move(options);
		return *this;
	}
	/// @copydoc program_t<cuda_cpp>::clear_options()
	program_t& clear_options() { options_ = {}; return *this; }

public:

	// TODO: Support specifying all compilation option in a single string and parsing it

	/// @copydoc program_t<cuda_cpp>::compile()
	compilation_output_t<ptx> compile() const
	{
		if (source_ == nullptr or *source_ == '\0') {
			throw ::std::invalid_argument("Attempt to compile a CUDA program without any source code");
		}
		auto marshalled_options = detail_::marshal(options_);
		::std::vector<const char*> option_ptrs = marshalled_options.option_ptrs();
		return program::detail_::compile_ptx(
			name_.c_str(),
			source_,
			{option_ptrs.data(), option_ptrs.size()});
	}

public: // constructors and destructor
	program_t(::std::string name) : parent(std::move(name)) {}
	program_t(const program_t&) = default;
	program_t(program_t&&) = default;
	~program_t() = default;

public: // operators

	///@cond
	program_t& operator=(const program_t& other) = default;
	program_t& operator=(program_t&& other) = default;
	///@endcond
}; // class program_t<ptx>

#endif // CUDA_VERSION >= 11010

namespace program {

/**
 * Create a new (not-yet-compiled) program without setting most of its
 * constituent fields.
 */
template <source_kind_t Kind>
inline program_t<Kind> create(const char* program_name)
{
	return program_t<Kind>(program_name);
}

/// @copydoc create <source_kind_t>(const char*)
template <source_kind_t Kind>
inline program_t<Kind> create(const ::std::string& program_name)
{
	return program_t<Kind>(program_name);
}

} // namespace program

/**
 * @returns all compute capabilities supported as targets by NVRTC and (most likely)
 * also by the PTX compilation library.
 *
 * @note the compute capabilities are returned in ascending order.
 */
#if CUDA_VERSION >= 11020
inline unique_span<device::compute_capability_t>
supported_targets()
{
	int num_supported_archs;
	auto status = nvrtcGetNumSupportedArchs(&num_supported_archs);
	throw_if_error<cuda_cpp>(status, "Failed obtaining the number of target NVRTC architectures");
	auto raw_archs = ::std::unique_ptr<int[]>(new int[num_supported_archs]);
	status = nvrtcGetSupportedArchs(raw_archs.get());
	throw_if_error<cuda_cpp>(status, "Failed obtaining the architectures supported by NVRTC");
	auto result = make_unique_span<device::compute_capability_t>(num_supported_archs);
	::std::transform(raw_archs.get(), raw_archs.get() + num_supported_archs, ::std::begin(result),
		[](int raw_arch) { return device::compute_capability_t::from_combined_number(raw_arch); });
	return result;
}
#endif

} // namespace rtc

} // namespace cuda

#endif // CUDA_API_WRAPPERS_RTC_PROGRAM_HPP_
