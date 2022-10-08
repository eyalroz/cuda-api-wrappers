/**
 * @file
 *
 * @brief The name_caching_progam class
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NVRTC_NAME_CACHING_PROGRAM_HPP_
#define CUDA_API_WRAPPERS_NVRTC_NAME_CACHING_PROGRAM_HPP_

#include <cuda/nvrtc/program.hpp>

#include <algorithm>
#include <cstring>

namespace cuda {

namespace detail_ {

template<typename C>
typename C::iterator insert_in_order(C& sorted_container, typename C::value_type&& item )
{
	auto pos = ::std::upper_bound( ::std::begin(sorted_container), ::std::end(sorted_container), item);
	return sorted_container.insert(pos, item);
}

namespace rtc {


/**
 * A dynamically-compilable program which caches mangled names
 * of kernels, so that the class user does not itself need to specify them
 *
 * @inherit program_t
 *
 * @todo Use a vector-backed map
 */
class name_caching_program_t : public program_t {
public:
	using parent = program_t;

public: // non-mutators

	const char* get_mangled_registered_name(const char* unmangled_name) const
	{
		auto comparator = [](const char* lhs, const char* rhs) { return ::std::strcmp(lhs, rhs) == 0};
		auto search_result = ::std::lower_bound(names_.cbegin(), names_.cend(), unmangled_name);
		if (search_result == names_.cend() {
			throw ::std::invalid_argument("Unmangled name " + unmangled_name + " was not registered for program " + name_);
		}
		auto& mangled = mangled_names[search_result - names_.cbegin()];
		if (mangled == nullptr) {
			mangled = parent::get_mangled_registered_name(unmangled_name);
		}
		return mangled;
	}

public: // mutators of the program, but not of this wrapper class
	void register_name_for_lookup(const char* unmangled_name)
	{
		detail_::insert_in_order(names_.unmangled_name);
		parent::register_name_for_lookup(unmangled_name);
	}

protected: // constructors
	name_caching_program_t(
		program::handle_t handle,
		const char* name,
		bool owning = false) : program_t(handle, name, owning) {}

public: // constructors and destructor

	name_caching_program_t(
		const char*  program_name,
		const char*  cuda_source,
		size_t       num_headers,
		const char** header_names,
		const char** header_sources
		) : program_t(program_name, cuda_source, num_headers, header_names, header_sources)
	{ }

	name_caching_program_t(const name_caching_program_t&) = delete;

	name_caching_program_t(name_caching_program_t&& other) noexcept
		: program_t(other), names_(other.names_), mangled_names(other.mangled_names_)
	{ };

	~name_caching_program_t() = default;

public: // operators

	name_caching_program_t& operator=(const name_caching_program_t& other) = delete;
	name_caching_program_t& operator=(name_caching_program_t&& other) = delete;

protected: // data members

	// These two are kept sorted
	// TODO: Use a single vector of pairs?
	::std::vector<const char*> names_ {};
	::std::vector<const char*> mangled_names {};

}; // class program_t

namespace program {

// template <>
inline program_t create(
	const char* program_name,
	const char* cuda_source,
	size_t num_headers,
	const char** header_names,
	const char** header_sources)
{
	return program_t(program_name, cuda_source, num_headers, header_names, header_sources);

}


template <typename HeaderNamesFwdIter, typename HeaderSourcesFwdIter>
inline program_t create(
	const char* program_name,
	const char* cuda_source,
	HeaderNamesFwdIter header_names_start,
	HeaderNamesFwdIter header_names_end,
	HeaderSourcesFwdIter header_sources_start)
{
	auto num_headers = header_names_end - header_names_start;
	::std::vector<const char*> header_names;
	header_names.reserve(num_headers);
	::std::copy_n(header_names_start, num_headers, ::std::back_inserter(header_names));
	::std::vector<const char*> header_sources;
	header_names.reserve(num_headers);
	::std::copy_n(header_sources_start, num_headers, ::std::back_inserter(header_sources));
	return program_t(cuda_source, program_name, num_headers, header_names.data(), header_sources.data());
}

inline program_t create(
	const char* program_name,
	const char* cuda_source,
	span<const char*> header_names,
	span<const char*> header_sources)
{
	return create (
		program_name,
		cuda_source,
		header_names.size(),
		header_names.data(),
		header_sources.data()
	);
}

/**
 * Create a run-time-compiled CUDA program using just a source string
 * with no extra headers
 */
inline program_t create(const char* program_name, const char* cuda_source)
{
	return create(program_name, cuda_source, 0, nullptr, nullptr);
}

template <typename HeaderNameAndSourceFwdIter>
inline program_t create(
	const char* program_name,
	const char* cuda_source,
	HeaderNameAndSourceFwdIter headers_start,
	HeaderNameAndSourceFwdIter headers_end)
{
	auto num_headers = headers_end - headers_start;
	::std::vector<const char*> header_names{};
	::std::vector<const char*> header_sources{};
	header_names.reserve(num_headers);
	header_sources.reserve(num_headers);
	for(auto& it = headers_start; it < headers_end; it++) {
		header_names.push_back(it->first);
		header_sources.push_back(it->second);
	}
	return create(cuda_source, program_name, num_headers, header_names.data(), header_sources.data());
}

// Note: This won't work for a string->string map... and we can't use a const char* to const char* map, I think.
template <typename HeaderNameAndSourceContainer>
inline program_t create(
	const char*                   program_name,
	const char*                   cuda_source,
	HeaderNameAndSourceContainer  headers)
{
	return create(cuda_source, program_name, headers.cbegin(), headers.cend());
}

namespace detail_ {

inline ::std::string identify(const program_t& program)
{
	return identify(program.handle(), program.name().c_str());
}

} // namespace detail_

} // namespace program

} // namespace rtc

///@cond
class module_t;
///@endcond
namespace module {

inline module_t create(
	const context_t&        context,
	const rtc::program_t&   compiled_program,
	const link::options_t&  options = {} )
{
#if CUDA_VERSION >= 11030
	auto cubin = compiled_program.cubin();
	return module::create(context, cubin.data(), options);
#else
	// Note this is less likely to succeed :-(
	auto ptx = compiled_program.ptx();
	return module::create(context, ptx.data(), options);
#endif
}

} // namespace module

} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVRTC_NAME_CACHING_PROGRAM_HPP_
