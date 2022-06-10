/**
 * @file
 *
 * @brief A Wrapper class for runtime-compiled (RTC) programs, manipulated using the NVRTC library.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NVRTC_PROGRAM_HPP_
#define CUDA_API_WRAPPERS_NVRTC_PROGRAM_HPP_

#include <cuda/nvrtc/compilation_options.hpp>
#include <cuda/nvrtc/error.hpp>
#include <cuda/nvrtc/types.hpp>
#include <cuda/api.hpp>

#include <vector>
#include <iostream>

namespace cuda {

///@cond
class device_t;
class context_t;

namespace rtc {
class program_t;
} // namespace rtc

namespace link {
struct options_t;
} // namespace rtc

namespace device {

class primary_context_t;

} // namespace device

///@endcond


///@cond
class module_t;
///@endcond
namespace module {

inline module_t create(
	const context_t&       context,
	const rtc::program_t&  compiled_program,
	link::options_t        options = {});

inline module_t create(
	device_t&              device,
	const rtc::program_t&  compiled_program,
	link::options_t        options = {});

} // namespace module

/**
 * @brief Real-time compilation of CUDA programs using the NVIDIA NVRTC library.
 */
namespace rtc {

///@cond
class program_t;
///@endcond

namespace program {

using handle_t = nvrtcProgram;

namespace detail_ {

inline ::std::string identify(handle_t handle)
{
	return "program at " + cuda::detail_::ptr_as_hex(handle);
}

inline ::std::string identify(handle_t handle, const char* name)
{
	return ::std::string("program ") + name + " at " + cuda::detail_::ptr_as_hex(handle);
}

::std::string identify(const program_t& program);

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
	program::handle_t handle() const { return handle_; }

public: // non-mutators

	// Unfortunately, C++'s standard string class is very inflexible,
	// and it is not possible for us to get it to have an appropriately-
	// sized _uninitialized_ buffer. We will therefore have to use
	// a clunkier return type.
	//
	// ::std::string log() const

	/**
	 * Obtain a copy of the log of the last compilation
	 *
	 * @note This will fail if the program has never been compiled.
	 */
	dynarray<char> compilation_log() const
	{
		size_t size;
		auto status = nvrtcGetProgramLogSize(handle_, &size);
		throw_if_error(status, "Failed obtaining compilation log size for " + program::detail_::identify(*this));
		dynarray<char> result(size);
		status = nvrtcGetProgramLog(handle_, result.data());
		throw_if_error(status, "Failed obtaining compilation log for" + program::detail_::identify(*this));
		return result;
	}

	/**
	 * Obtain a copy of the PTX result of the last compilation.
	 *
	 * @note the PTX may be missing in cases such as compilation failure or link-time
	 * optimization compilation.
	 * @note This will fail if the program has never been compiled.
	 */
	dynarray<char> ptx() const
	{
		size_t size;
		auto status = nvrtcGetPTXSize(handle_, &size);
		throw_if_error(status, "Failed obtaining NVRTC program output PTX size for "
			+ cuda::rtc::program::detail_::identify(*this));
		dynarray<char> result(size);
		status = nvrtcGetPTX(handle_, result.data());
		throw_if_error(status, "Failed obtaining NVRTC program output PTX for "
			+ cuda::rtc::program::detail_::identify(*this));
		return result;
	}

	bool has_ptx() const
	{
		size_t size;
		auto status = nvrtcGetPTXSize(handle_, &size);
		if (status == NVRTC_ERROR_INVALID_PROGRAM) { return false; }
		throw_if_error(status, "Failed determining whether the NVRTC program has a compiled PTX result: " +
			cuda::rtc::program::detail_::identify(*this));
		if (size == 0) {
			throw ::std::logic_error("PTX size reported as 0 by NVRTC for program: " +
				cuda::rtc::program::detail_::identify(*this));
		}
		return true;
	}

#if CUDA_VERSION >= 11010
	/**
	 * Obtain a copy of the CUBIN result of the last compilation.
	 *
	 * @note CUBIN output is not available when compiling for a virtual architecture only.
	 * Also, it may be missing in cases such as compilation failure or link-time
	 * optimization compilation.
	 * @note This will fail if the program has never been compiled.
	 */
	dynarray<char> cubin() const
	{
		size_t size;
		auto status = nvrtcGetCUBINSize(handle_, &size);
		throw_if_error(status, "Failed obtaining NVRTC program output CUBIN size");
		if (size == 0) {
			throw ::std::invalid_argument("CUBIN requested for a CUDA program compiled for a virtual architecture: " +
				cuda::rtc::program::detail_::identify(*this));
		}
		dynarray<char> result(size);
		status = nvrtcGetCUBIN(handle_, result.data());
		throw_if_error(status, "Failed obtaining NVRTC program output CUBIN for"
			+ cuda::rtc::program::detail_::identify(*this));
		return result;
	}

	bool has_cubin() const
	{
		size_t size;
		auto status = nvrtcGetCUBINSize(handle_, &size);
		if (status == NVRTC_ERROR_INVALID_PROGRAM) { return false; }
		throw_if_error(status, "Failed determining whether the NVRTC program has a compiled CUBIN result: " +
			cuda::rtc::program::detail_::identify(*this));
		return (size > 0);
	}

#endif

#if CUDA_VERSION >= 11040
	/**
	 * Obtain a copy of the nvvm intermediate format result of the last compilation
	 */
	dynarray<char> nvvm() const
	{
		size_t size;
		auto status = nvrtcGetNVVMSize(handle_, &size);
		throw_if_error(status, "Failed obtaining NVRTC program output NVVM size");
		dynarray<char> result(size);
		status = nvrtcGetNVVM(handle_, result.data());
		throw_if_error(status, "Failed obtaining NVRTC program output NVVM");
		return result;
	}

	bool has_nvvm() const
	{
		size_t size;
		auto status = nvrtcGetNVVMSize(handle_, &size);
		if (status == NVRTC_ERROR_INVALID_PROGRAM) { return false; }
		throw_if_error(status, "Failed determining whether the NVRTC program has a compiled NVVM result: " +
			cuda::rtc::program::detail_::identify(*this));
		if (size == 0) {
			throw ::std::logic_error("NVVM size reported as 0 by NVRTC for program: " +
				cuda::rtc::program::detail_::identify(*this));
		}
		return true;
	}

#endif

	/**
	 * Obtain the mangled/lowered form of an expression registered earlier, after
	 * the compilation.
	 *
	 * @param unmangled_name A name of a __global__ or __device__ function or variable.
	 * @return The mangled name (which can actually be used for invoking kernels,
	 * moving data etc.). The memory is owned by the NVRTC program and will be
	 * released when it is destroyed.
	 */
	const char* get_mangling_of(const char* unmangled_name) const
	{
		const char* result;
		auto status = nvrtcGetLoweredName(handle_, unmangled_name, &result);
		throw_if_error(status, ::std::string("Failed obtaining the mangled form of name \"")
			+ unmangled_name + "\" in dynamically-compiled program \"" + name_ + '\"');
		return result;
	}

	const char* get_mangling_of(const ::std::string& unmangled_name) const
	{
		return get_mangling_of(unmangled_name.c_str());
	}

public: // mutators of the program, but not of this wrapper class
	void compile(span<const char*> raw_options) const
	{
		auto status = nvrtcCompileProgram(handle_, (int) raw_options.size(), raw_options.data());
		throw_if_error(status, "Failed compiling program \"" + name_ + "\"");
	}

	// TODO: Support specifying all compilation option in a single string and parsing it

	void compile(const compilation_options_t& options = compilation_options_t{}) const
	{
		auto marshalled_options = marshal(options);
		auto option_ptrs = marshalled_options.option_ptrs();
		compile({option_ptrs.data(), option_ptrs. size()});
	}

	void compile_for(
		device::compute_capability_t target_compute_capability,
		compilation_options_t extra_options = compilation_options_t{}) const
	{
		extra_options.set_target(target_compute_capability);
		return compile(extra_options);
	}

	void compile_for(
		device_t               device,
		compilation_options_t  extra_options = compilation_options_t{}) const
	{
		compile_for(device.compute_capability(), extra_options);
	}

	void compile_for(
		const context_t        context,
		compilation_options_t  extra_options = compilation_options_t{}) const
	{
		return compile_for(context.device(), extra_options);
	}

	/**
	 * @brief Register a pre-mangled name of a kernel, to make available for use
	 * after compilation
	 *
	 * @param name The text of an expression, e.g. "my_global_func()", "f1", "N1::N2::n2",
	 *
	 */
	void register_global(const char* unmangled_name)
	{
		auto status = nvrtcAddNameExpression(handle_, unmangled_name);
		throw_if_error(status, "Failed registering a mangled name with program \"" + name_ + "\"");
	}

	void register_global(const ::std::string& unmanged_name)
	{
		register_global(unmanged_name.c_str());
	}

	template <typename StringType1, typename StringType2, typename... StringTypes>
	void register_globals(StringType1 first, StringType2 second, StringTypes&&... more_globals)
	{
		register_global(first);
		register_global(second);
		cuda::detail_::for_each_argument(
			[this](::std::string global_name) { register_global(global_name); },
			more_globals...);
	}

	template <typename Container>
	void register_globals(Container&& names)
	{
		for(const auto& name : names) {
			register_global(name);
		};
	}

protected: // constructors
	program_t(
		program::handle_t handle,
		const char* name,
		bool owning = false) : handle_(handle), name_(name), owning_(owning) { }

public: // constructors and destructor

	program_t(
		const char*  program_name,
		const char*  cuda_source,
		size_t       num_headers,
		const char** header_names,
		const char** header_sources
		) : handle_(), name_(program_name), owning_(true)
	{
		status_t status;
#ifndef NDEBUG
		if (num_headers > ::std::numeric_limits<int>::max()) {
			throw ::std::invalid_argument("Cannot process more than " + ::std::to_string(::std::numeric_limits<int>::max()) + " headers.");
		}
#endif
		status = nvrtcCreateProgram(&handle_, cuda_source, program_name, (int) num_headers, header_sources, header_names);
		throw_if_error(status, "Failed creating an NVRTC program (named " + ::std::string(name_) + ')');
	}

	program_t(const program_t&) = delete;

	program_t(program_t&& other) noexcept
		: handle_(other.handle_), name_(::std::move(other.name_)), owning_(other.owning_)
	{
		other.owning_ = false;
	};

	~program_t()
	{
		if (owning_) {
			auto status = nvrtcDestroyProgram(&handle_);
			throw_if_error(status, "Destroying an NVRTC program");
		}
	}

public: // operators

	program_t& operator=(const program_t& other) = delete;
	program_t& operator=(program_t&& other) = delete;

protected: // data members
	program::handle_t  handle_;
	::std::string        name_;
	bool               owning_;
}; // class program_t

namespace program {
inline program_t create(
	const char* program_name,
	const char* cuda_source,
	span<const char*> header_names,
	span<const char*> header_sources)
{
	return program_t{ program_name, cuda_source, header_names.size(), header_names.data(), header_sources.data() };
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
	return program_t{ cuda_source, program_name, num_headers, header_names.data(), header_sources.data() };
}

template <typename HeaderNameAndSourceFwdIter>
inline program_t create(
	const char* program_name,
	const char* cuda_source,
	HeaderNameAndSourceFwdIter named_headers_start,
	HeaderNameAndSourceFwdIter named_headers_end)
{
	auto num_headers = named_headers_end - named_headers_start;
	::std::vector<const char*> header_names{};
	::std::vector<const char*> header_sources{};
	header_names.reserve(num_headers);
	header_sources.reserve(num_headers);
	for(auto& it = named_headers_start; it < named_headers_end; it++) {
		header_names.push_back(it->first);
		header_sources.push_back(it->second);
	}
	return program_t{ program_name, cuda_source, (size_t) num_headers, header_names.data(), header_sources.data() };
}

// Note: This won't work for a string->string map... and we can't use a const char* to const char* map, I think.
template <typename HeaderNameAndSourcePairContainer>
inline program_t create(
	const char*                       program_name,
	const char*                       cuda_source,
	HeaderNameAndSourcePairContainer  named_headers)
{
	return create<typename HeaderNameAndSourcePairContainer::const_iterator>(
		program_name, cuda_source, named_headers.cbegin(), named_headers.cend());
}

/**
 * Create a run-time-compiled CUDA program using just a source string
 * with no extra headers
 */
inline program_t create(const char* program_name, const char* cuda_source)
{
	return program_t{ program_name, cuda_source, 0, nullptr, nullptr };
}

/**
 * Creates a program with no main source code made available as a string.
 *
 * @note this allows later inclusion of all source code from files via
 * the pre-include mechanism of NVRTC.
 *
 * @return The created empty-source program
 */
inline program_t create_empty(const char* program_name)
{
	return program_t{ program_name, "", 0, nullptr, nullptr };
}

namespace detail_ {

inline ::std::string identify(const program_t& program)
{
	return identify(program.handle(), program.name().c_str());
}

} // namespace detail_

} // namespace program

} // namespace rtc

namespace module {

inline module_t create(
	const context_t&       context,
	const rtc::program_t&  compiled_program,
	link::options_t        options)
{
#if CUDA_VERSION >= 11030
	auto cubin = compiled_program.cubin();
	return module::create(context, cubin, options);
#else
	// Note this is less likely to succeed :-(
	auto ptx = compiled_program.ptx();
	return module::create(context, ptx, options);
#endif
}

inline module_t create(
	device_t&              device,
	const rtc::program_t&  compiled_program,
	link::options_t        options)
{
	return create(device.primary_context(), compiled_program, options);
}

} // namespace module

} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVRTC_PROGRAM_HPP_
