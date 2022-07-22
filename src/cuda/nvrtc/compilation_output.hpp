/**
 * @file
 *
 * @brief Contains the @ref cuda::rtc::compilation_output_t class and related code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NVRTC_OUTPUT_HPP_
#define CUDA_API_WRAPPERS_NVRTC_OUTPUT_HPP_

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
class compilation_output_t;
} // namespace rtc;

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
	const context_t&                  context,
	const rtc::compilation_output_t&  compiled_program,
	link::options_t                   options = {});

inline module_t create(
	device_t&                         device,
	const rtc::compilation_output_t&  compiled_program,
	link::options_t                   options = {});

} // namespace module

/**
 * @brief Real-time compilation of CUDA programs using the NVIDIA NVRTC library.
 */
namespace rtc {

namespace program {
namespace detail_ {

inline ::std::string identify(program::handle_t handle, const char *name = nullptr)
{
	return ::std::string("program ") + (name == nullptr ? "" : name) + " at " + cuda::detail_::ptr_as_hex(handle);
}

} // namespace detail_

#if CUDA_VERSION >= 11040
namespace detail_ {

size_t get_nvvm_size(program::handle_t program_handle, const char *program_name = nullptr)
{
	size_t size;
	auto status = nvrtcGetNVVMSize(program_handle, &size);
	throw_if_error(status, "Failed obtaining NVRTC program output NVVM size for "
		+ identify(program_handle, program_name));
	return size;
}

void get_nvvm(char* buffer, program::handle_t program_handle, const char *program_name = nullptr)
{
	auto status = nvrtcGetNVVM(program_handle, buffer);
	throw_if_error(status, "Failed obtaining NVRTC program output NVVM for "
						   + identify(program_handle, program_name));

}

} // namespace detail_
#endif // CUDA_VERSION >= 11040

} // namespace program

namespace compilation_output {

namespace detail_ {

::std::string identify(const compilation_output_t &compilation_output);

inline compilation_output_t wrap(
	program::handle_t  program_handle,
	::std::string      program_name,
	bool               succeeded,
	bool               own_handle);

} // namespace detail

} // namespace compilation_output

/**
 * Wrapper class for the result of an NVRTC compilation (including the program handle) -
 * whether it succeeded or failed due to errors in the program itself.
 *
 * @note This class _may_ own an NVRTC low-level program handle.
 * @note If compilation failed due to apriori-invalid arguments - an exception will
 * have been thrown. The only failure this class may represent
 */
class compilation_output_t {

public: // getters

	bool succeeded() const { return succeeded_; }
	operator bool() const { return succeeded_; }
	const ::std::string& program_name() const { return program_name_; }
	program::handle_t program_handle() const { return program_handle_; }

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
	dynarray<char> log() const
	{
		size_t size;
		auto status = nvrtcGetProgramLogSize(program_handle_, &size);
		throw_if_error(status, "Failed obtaining compilation log size for "
			+ compilation_output::detail_::identify(*this));
		dynarray<char> result(size);
		status = nvrtcGetProgramLog(program_handle_, result.data());
		throw_if_error(status, "Failed obtaining compilation log for"
			+ compilation_output::detail_::identify(*this));
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
		auto status = nvrtcGetPTXSize(program_handle_, &size);
		throw_if_error(status, "Failed obtaining output PTX size for "
			+ compilation_output::detail_::identify(*this));
		dynarray<char> result(size);
		status = nvrtcGetPTX(program_handle_, result.data());
		throw_if_error(status, "Failed obtaining output PTX for "
			+ compilation_output::detail_::identify(*this));
		return result;
	}

	bool has_ptx() const
	{
		size_t size;
		auto status = nvrtcGetPTXSize(program_handle_, &size);
		if (status == NVRTC_ERROR_INVALID_PROGRAM) { return false; }
		throw_if_error(status, "Failed determining whether the NVRTC program has a compiled PTX result: "
			+ compilation_output::detail_::identify(*this));
		if (size == 0) {
			throw ::std::logic_error("PTX size reported as 0 by "
				+ compilation_output::detail_::identify(*this));
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
		auto status = nvrtcGetCUBINSize(program_handle_, &size);
		throw_if_error(status, "Failed obtaining NVRTC program output CUBIN size");
		if (size == 0) {
			throw ::std::invalid_argument("CUBIN requested for a program compiled for a virtual architecture only: "
				+ compilation_output::detail_::identify(*this));
		}
		dynarray<char> result(size);
		status = nvrtcGetCUBIN(program_handle_, result.data());
		throw_if_error(status, "Failed obtaining NVRTC program output CUBIN for "
			+ compilation_output::detail_::identify(*this));
		return result;
	}

	bool has_cubin() const
	{
		size_t size;
		auto status = nvrtcGetCUBINSize(program_handle_, &size);
		if (status == NVRTC_ERROR_INVALID_PROGRAM) { return false; }
		throw_if_error(status, "Failed determining whether the NVRTC program has a compiled CUBIN result: "
			+ compilation_output::detail_::identify(*this));
		return (size > 0);
	}

#endif

#if CUDA_VERSION >= 11040
	/**
	 * Obtain a copy of the nvvm intermediate format result of the last compilation
	 *
	 * @throws ::std::invalid_argument if the supplied buffer is too small to hold
	 * the program's NVVM.
	 *
	 * @param[inout] buffer A writable buffer which should, hopefully, be large
	 * enough to contain the compiled program's NVVM
	 * @return The sub-buffer, starting at the beginning of @p buffer, containing
	 * exactly the compiled program's NVVM (i.e. sized down to fit the contents)
	 */
	/// @{
	span<char> nvvm(span<char> buffer) const
	{
		size_t size = program::detail_::get_nvvm_size(program_handle_, program_name_.c_str());
		if (buffer.size() < size) {
			throw ::std::invalid_argument("Provided buffer size is insufficient for the compiled program's NVVM ("
				+ ::std::to_string(buffer.size()) + " < " + ::std::to_string(size) + ": "
				+ compilation_output::detail_::identify(*this));
		}
		program::detail_::get_nvvm(buffer.data(), program_handle_, program_name_.c_str());
		return { buffer.data(), size };
	}

	dynarray<char> nvvm() const
	{
		size_t size = program::detail_::get_nvvm_size(program_handle_, program_name_.c_str());
		dynarray<char> result(size);
		program::detail_::get_nvvm(result.data(), program_handle_, program_name_.c_str());
		return result;
	}
	/// @}

	bool has_nvvm() const
	{
		size_t size;
		auto status = nvrtcGetNVVMSize(program_handle_, &size);
		if (status == NVRTC_ERROR_INVALID_PROGRAM) { return false; }
		throw_if_error(status, "Failed determining whether the NVRTC program has a compiled NVVM result: "
			+ compilation_output::detail_::identify(*this));
		if (size == 0) {
			throw ::std::logic_error("NVVM size reported as 0 by NVRTC for program: "
				+ compilation_output::detail_::identify(*this));
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
		auto status = nvrtcGetLoweredName(program_handle_, unmangled_name, &result);
		throw_if_error(status, ::std::string("Failed obtaining the mangled form of name \"")
			+ unmangled_name + "\" in dynamically-compiled program \"" + program_name_ + '\"');
		return result;
	}

	const char* get_mangling_of(const ::std::string& unmangled_name) const
	{
		return get_mangling_of(unmangled_name.c_str());
	}


protected: // constructors
	compilation_output_t(program::handle_t handle, ::std::string name, bool succeeded, bool owning = false)
	: program_handle_(handle), program_name_(name), succeeded_(succeeded), owns_handle_(owning) { }

	friend compilation_output_t compilation_output::detail_::wrap(
		program::handle_t  program_handle,
		::std::string      program_name,
		bool               succeeded,
		bool               own_handle);

public:

	compilation_output_t(const compilation_output_t& other)
	:
		program_handle_(other.program_handle_),
		program_name_(other.program_name_),
		owns_handle_(false)
	{
	}

	compilation_output_t(compilation_output_t&& other) noexcept
	:
		program_handle_(other.program_handle_),
		program_name_(::std::move(other.program_name_)),
		owns_handle_(other.owns_handle_)
	{
		other.owns_handle_ = false;
	};

	~compilation_output_t()
	{
		if (owns_handle_) {
			auto status = nvrtcDestroyProgram(&program_handle_);
			throw_if_error(status, "Destroying " + program::detail_::identify(program_handle_, program_name_.c_str()));
		}
	}

public: // operators

	compilation_output_t& operator=(const compilation_output_t& other) = delete;
	compilation_output_t& operator=(compilation_output_t&& other) = delete;

protected: // data members
	program::handle_t  program_handle_;
	::std::string      program_name_;
	bool               succeeded_;
	bool               owns_handle_;
}; // class compilation_output_t

namespace compilation_output {

namespace detail_ {

inline ::std::string identify(const compilation_output_t &compilation_result)
{
	return "Compilation output of " +
		program::detail_::identify(compilation_result.program_handle(), compilation_result.program_name().c_str());
}

inline compilation_output_t wrap(
	program::handle_t  program_handle,
	::std::string      program_name,
	bool               succeeded,
	bool               own_handle)
{
	return compilation_output_t{program_handle, ::std::move(program_name), succeeded, own_handle};
}

} // namespace detail_

} // namespace compilation_output

} // namespace rtc

namespace module {

inline module_t create(
	const context_t&                  context,
	const rtc::compilation_output_t&  compiled_program,
	link::options_t                   options)
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
	device_t&                         device,
	const rtc::compilation_output_t&  compiled_program,
	link::options_t                   options)
{
	return create(device.primary_context(), compiled_program, options);
}

} // namespace module

} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVRTC_OUTPUT_HPP_
