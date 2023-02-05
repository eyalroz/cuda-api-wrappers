/**
 * @file
 *
 * @brief Contains the @ref cuda::rtc::compilation_output_t class and related code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NVRTC_OUTPUT_HPP_
#define CUDA_API_WRAPPERS_NVRTC_OUTPUT_HPP_

#include <cuda/rtc/compilation_options.hpp>
#include <cuda/rtc/error.hpp>
#include <cuda/rtc/types.hpp>
#include <cuda/api.hpp>

#include <vector>
#include <iostream>

namespace cuda {

///@cond
class device_t;
class context_t;

namespace rtc {
template <source_kind_t Kind>
class compilation_output_t;
} // namespace rtc

namespace link {
struct options_t;
} // namespace rtc

namespace device {
class primary_context_t;
} // namespace device

class module_t;
///@endcond
namespace module {

template <source_kind_t Kind>
inline module_t create(
	const context_t&                        context,
	const rtc::compilation_output_t<Kind>&  compiled_program,
	const link::options_t&                  options = {});

template <source_kind_t Kind>
inline module_t create(
	device_t&                               device,
	const rtc::compilation_output_t<Kind>&  compiled_program,
	const link::options_t&                  options = {});

} // namespace module

/**
 * @brief Real-time compilation of CUDA programs using the NVIDIA NVRTC library.
 */
namespace rtc {

namespace program {

namespace detail_ {

template <source_kind_t Kind>
inline ::std::string identify(const char *name)
{
	return ::std::string{detail_::kind_name(Kind)} + " program" +
		((name == nullptr) ? "" : " '" + ::std::string{name} + "'");
}

template <source_kind_t Kind>
inline ::std::string identify(program::handle_t<Kind> handle, const char *name = nullptr)
{
	return identify<Kind>(name) + " at " + cuda::detail_::ptr_as_hex(handle);
}

template <source_kind_t Kind>
inline size_t get_log_size(program::handle_t<Kind> program_handle, const char* program_name)
{
	size_t size;
	auto status =
#if CUDA_VERSION >= 11010
		(Kind == ptx) ?
			(status_t<Kind>) nvPTXCompilerGetErrorLogSize((handle_t<ptx>) program_handle, &size) :
#endif // CUDA_VERSION >= 11010
			(status_t<Kind>) nvrtcGetProgramLogSize((handle_t<cuda_cpp>) program_handle, &size);
	throw_if_error<Kind>(status, "Failed obtaining compilation log size for "
		+ identify<Kind>(program_handle, program_name));
	return size;
}

template <source_kind_t Kind>
inline void get_log(char* buffer, program::handle_t<Kind> program_handle, const char *program_name = nullptr)
{
	auto status =
#if CUDA_VERSION >= 11010
		(Kind == ptx) ?
			(status_t<Kind>) nvPTXCompilerGetErrorLog((handle_t<ptx>)program_handle, buffer) :
#endif
			(status_t<Kind>) nvrtcGetProgramLog((handle_t<cuda_cpp>)program_handle, buffer);
	throw_if_error<Kind>(status, "Failed obtaining compilation log for "
		+ identify<Kind>(program_handle, program_name));
}

#if CUDA_VERSION >= 11010
template <source_kind_t Kind>
inline size_t get_cubin_size(program::handle_t<Kind> program_handle, const char* program_name)
{
	size_t size;
	status_t<Kind> status = (Kind == cuda_cpp) ?
		(status_t<Kind>) nvrtcGetCUBINSize((program::handle_t<cuda_cpp>) program_handle, &size) :
		(status_t<Kind>) nvPTXCompilerGetCompiledProgramSize((program::handle_t<ptx>) program_handle, &size);
	throw_if_error<Kind>(status, "Failed obtaining program output CUBIN size for "
		+ identify<Kind>(program_handle, program_name));
	if (size == 0) {
		throw  (Kind == cuda_cpp) ?
			::std::runtime_error("Output CUBIN requested for a compilation for a virtual architecture only of "
				+ identify<Kind>(program_handle, program_name)):
			::std::runtime_error("Empty output CUBIN for compilation of "
				+ identify<Kind>(program_handle, program_name));
	}
	return size;
}

template <source_kind_t Kind>
inline void get_cubin(char* buffer, program::handle_t<Kind> program_handle, const char *program_name = nullptr)
{
	status_t<Kind> status = (Kind == cuda_cpp) ?
		(status_t<Kind>) nvrtcGetCUBIN((program::handle_t<cuda_cpp>) program_handle, buffer) :
		(status_t<Kind>) nvPTXCompilerGetCompiledProgram((program::handle_t<ptx>) program_handle, buffer);
	throw_if_error<Kind>(status, "Failed obtaining compilation output CUBIN for "
		  + identify<Kind>(program_handle, program_name));
}
#endif // CUDA_VERSION >= 11010

inline size_t get_ptx_size(program::handle_t<cuda_cpp> program_handle, const char *program_name = nullptr)
{
	size_t size;
	auto status = nvrtcGetPTXSize(program_handle, &size);
	throw_if_error<cuda_cpp>(status, "Failed obtaining compilation output PTX size for compilation of "
		+ identify<cuda_cpp>(program_handle, program_name));
	return size;
}

inline void get_ptx(char* buffer, program::handle_t<cuda_cpp> program_handle, const char *program_name = nullptr)
{
	auto status = nvrtcGetPTX(program_handle, buffer);
	throw_if_rtc_error_lazy(cuda_cpp, status, "Failed obtaining compilation output PTX for compilation of "
		+ identify<cuda_cpp>(program_handle, program_name));
}

#if CUDA_VERSION >= 11040

inline size_t get_nvvm_size(program::handle_t<cuda_cpp> program_handle, const char *program_name = nullptr)
{
	size_t size;
	auto status = nvrtcGetNVVMSize(program_handle, &size);
	throw_if_rtc_error_lazy(cuda_cpp, status, "Failed obtaining output NVVM size for compilation of "
		+ identify<cuda_cpp>(program_handle, program_name));
	return size;
}

inline void get_nvvm(char* buffer, program::handle_t<cuda_cpp> program_handle, const char *program_name = nullptr)
{
	auto status = nvrtcGetNVVM(program_handle, buffer);
	throw_if_rtc_error_lazy(cuda_cpp, status, "Failed obtaining output NVVM for compilation of "
		+ identify<cuda_cpp>(program_handle, program_name));
}

#endif // CUDA_VERSION >= 11040

} // namespace detail_

} // namespace program

namespace compilation_output {

namespace detail_ {

template <source_kind_t Kind>
::std::string identify(const compilation_output_t<Kind> &compilation_output);

template <source_kind_t Kind>
inline compilation_output_t<Kind> wrap(
	program::handle_t<Kind>  program_handle,
	::std::string            program_name,
	bool                     succeeded,
	bool                     own_handle);

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
template <source_kind_t Kind>
class compilation_output_base_t {
public: // types and constants
	constexpr static const source_kind_t source_kind { Kind };
	using handle_type = program::handle_t<source_kind>;
	using status_type = status_t<source_kind>;

public: // getters
	bool succeeded() const { return succeeded_; }
	bool failed() const { return not succeeded_; }
	operator bool() const { return succeeded_; }
	const ::std::string& program_name() const { return program_name_; }
	handle_type program_handle() const { return program_handle_; }

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
	///@{
	span<char> log(span<char> buffer) const
	{
		size_t size = program::detail_::get_log_size<source_kind>(program_handle_, program_name_.c_str());
		if (buffer.size() < size) {
			throw ::std::invalid_argument(
				"Provided buffer size is insufficient for the program compilation log ("
				+ ::std::to_string(buffer.size()) + " < " + ::std::to_string(size) + ": "
				+ compilation_output::detail_::identify(*this));
		}
		program::detail_::get_log(buffer.data(), program_handle_, program_name_.c_str());
		return { buffer.data(), size };
	}

	dynarray<char> log() const
	{
		size_t size = program::detail_::get_log_size<source_kind>(program_handle_, program_name_.c_str());
		dynarray<char> result(size);
		program::detail_::get_log<source_kind>(result.data(), program_handle_, program_name_.c_str());
		return result;
	}
	///@}

#if CUDA_VERSION >= 11010
	virtual dynarray<char> cubin() const = 0;
	virtual bool has_cubin() const = 0;
#endif

protected: // constructors
	compilation_output_base_t(handle_type handle, ::std::string name, bool succeeded, bool owning = false)
	: program_handle_(handle), program_name_(::std::move(name)), succeeded_(succeeded), owns_handle_(owning) { }

public: // constructors & destructor
	compilation_output_base_t(compilation_output_base_t&& other) noexcept :
		program_handle_(other.program_handle_),
		program_name_(::std::move(other.program_name_)),
		succeeded_(other.succeeded_),
		owns_handle_(other.owns_handle_)
	{
		other.owns_handle_ = false;
	};

	~compilation_output_base_t() noexcept(false)
	{
		if (owns_handle_) {
			auto status =
#if CUDA_VERSION >= 11010
				(Kind == ptx) ?
					(status_t<Kind>) nvPTXCompilerDestroy((program::handle_t<ptx>*) &program_handle_) :
#endif
					(status_t<Kind>) nvrtcDestroyProgram((program::handle_t<cuda_cpp>*) &program_handle_);
			throw_if_error<Kind>(status, "Destroying " + program::detail_::identify<Kind>(program_handle_, program_name_.c_str()));
		}
	}

public: // operators

	compilation_output_base_t& operator=(const compilation_output_base_t& other) = delete;
	compilation_output_base_t& operator=(compilation_output_base_t&& other) = delete;

protected: // data members
	program::handle_t<Kind>  program_handle_;
	::std::string            program_name_;
	bool                     succeeded_;
	bool                     owns_handle_;

};

template <>
class compilation_output_t<cuda_cpp> : public compilation_output_base_t<cuda_cpp> {
public:
	using parent = compilation_output_base_t<cuda_cpp>;
	using parent::parent;

	friend compilation_output_t compilation_output::detail_::wrap<source_kind>(
		handle_type    program_handle,
		::std::string  program_name,
		bool           succeeded,
		bool           own_handle);

public: // non-mutators
	/**
	 * Obtain a (nul-terminated) copy of the PTX result of the last compilation.
	 *
	 * @note the PTX may be missing in cases such as compilation failure or link-time
	 * optimization compilation.
	 * @note This will fail if the program has never been compiled.
	 */
	///@{
	
	/**
	 * @param[inout] buffer A writable buffer large enough to contain the compiled
	 *     program's PTX code.
	 */ 	
	span<char> ptx(span<char> buffer) const
	{
		size_t size = program::detail_::get_ptx_size(parent::program_handle_, program_name_.c_str());
		if (buffer.size() < size) {
			throw ::std::invalid_argument("Provided buffer size is insufficient for the compiled program's PTX ("
				+ ::std::to_string(buffer.size()) + " < " + ::std::to_string(size) + ": "
				+ compilation_output::detail_::identify(*this));
		}
		program::detail_::get_ptx(buffer.data(), program_handle_, program_name_.c_str());
		return { buffer.data(), size };
	}

	dynarray<char> ptx() const
	{
		size_t size = program::detail_::get_ptx_size(program_handle_, program_name_.c_str());
		dynarray<char> result(size);
		program::detail_::get_ptx(result.data(), program_handle_, program_name_.c_str());
		return result;
	}
	///@}

	bool has_ptx() const
	{
		size_t size;
		status_type status = nvrtcGetPTXSize(program_handle_, &size);
		if (status == NVRTC_ERROR_INVALID_PROGRAM) { return false; }
		throw_if_rtc_error_lazy(source_kind, status, "Failed determining whether compilation resulted in PTX code for "
			+ compilation_output::detail_::identify<source_kind>(*this));
		if (size == 0) {
			throw ::std::logic_error("PTX size reported as 0 by "
				+ compilation_output::detail_::identify<source_kind>(*this));
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
	///@{
	span<char> cubin(span<char> buffer) const
	{
		size_t size = program::detail_::get_cubin_size<source_kind>(program_handle_, program_name_.c_str());
		if (buffer.size() < size) {
			throw ::std::invalid_argument("Provided buffer size is insufficient for the compiled program's cubin ("
				+ ::std::to_string(buffer.size()) + " < " + ::std::to_string(size) + ": "
				+ compilation_output::detail_::identify(*this));
		}
		program::detail_::get_cubin<source_kind>(buffer.data(), program_handle_, program_name_.c_str());
		return { buffer.data(), size };
	}

	dynarray<char> cubin() const override
	{
		size_t size = program::detail_::get_cubin_size<source_kind>(program_handle_, program_name_.c_str());
		dynarray<char> result(size);
		program::detail_::get_cubin<source_kind>(result.data(), program_handle_, program_name_.c_str());
		return result;
	}
	///@}

	bool has_cubin() const override
	{
		size_t size;
		auto status = nvrtcGetCUBINSize(program_handle_, &size);
		if (status == NVRTC_ERROR_INVALID_PROGRAM) { return false; }
		throw_if_rtc_error_lazy(cuda_cpp, status, "Failed determining whether the program has a compiled CUBIN result: "
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
	 * @param[inout] buffer A writable buffer large enough to contain the compiled
	 *     program's NVVM code.
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
		throw_if_rtc_error_lazy(cuda_cpp, status, "Failed determining whether the NVRTC program has a compiled NVVM result: "
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
		throw_if_error<source_kind>(status, ::std::string("Failed obtaining the mangled form of name \"")
			+ unmangled_name + "\" in dynamically-compiled program \"" + program_name_ + '\"');
		return result;
	}

	const char* get_mangling_of(const ::std::string& unmangled_name) const
	{
		return get_mangling_of(unmangled_name.c_str());
	}
}; // class compilation_output_t<cuda_cpp>

#if CUDA_VERSION >= 11010

template <>
class compilation_output_t<ptx> : public compilation_output_base_t<ptx> {
public:
	using parent = compilation_output_base_t<ptx>;
	using parent::parent;

	friend compilation_output_t compilation_output::detail_::wrap<source_kind>(
		handle_type    program_handle,
		::std::string  program_name,
		bool           succeeded,
		bool           own_handle);

public: // non-mutators
	/**
	 * Obtain a copy of the CUBIN result of the last compilation.
	 *
	 * @note This will fail if the program has never been compiled.
	 */
	///@{
	span<char> cubin(span<char> buffer) const
	{
		size_t size = program::detail_::get_cubin_size<source_kind>(program_handle_, program_name_.c_str());
		if (buffer.size() < size) {
			throw ::std::invalid_argument("Provided buffer size is insufficient for the compiled program's cubin ("
				+ ::std::to_string(buffer.size()) + " < " + ::std::to_string(size) + ": "
				+ compilation_output::detail_::identify<source_kind>(*this));
		}
		program::detail_::get_cubin<source_kind>(buffer.data(), program_handle_, program_name_.c_str());
		return { buffer.data(), size };
	}

	dynarray<char> cubin() const override
	{
		size_t size = program::detail_::get_cubin_size<source_kind>(program_handle_, program_name_.c_str());
		dynarray<char> result(size);
		program::detail_::get_cubin<source_kind>(result.data(), program_handle_, program_name_.c_str());
		return result;
	}
	///@}

	bool has_cubin() const override
	{
		size_t size;
		auto status = nvPTXCompilerGetCompiledProgramSize(program_handle_, &size);
		if (status == NVPTXCOMPILE_ERROR_INVALID_INPUT) { return false; }
		throw_if_error<source_kind>(status, "Failed determining whether the program has a compiled CUBIN result: "
			+ compilation_output::detail_::identify(*this));
		return (size > 0);
	}
}; // class compilation_output_t<ptx>

#endif // CUDA_VERSION >= 11010

namespace compilation_output {

namespace detail_ {

template <source_kind_t Kind>
inline ::std::string identify(const compilation_output_t<Kind> &compilation_result)
{
	return "Compilation output of " + program::detail_::identify<Kind>(
		compilation_result.program_handle(),
		compilation_result.program_name().c_str());
}

template <source_kind_t Kind>
inline compilation_output_t<Kind> wrap(
	program::handle_t<Kind>  program_handle,
	::std::string            program_name,
	bool                     succeeded,
	bool                     own_handle)
{
	return compilation_output_t<Kind>{program_handle, ::std::move(program_name), succeeded, own_handle};
}

} // namespace detail_

} // namespace compilation_output

} // namespace rtc

namespace module {

template <source_kind_t Kind>
module_t create(
	const context_t&                        context,
	const rtc::compilation_output_t<Kind>&  compiled_program,
	const link::options_t&                  options);

template<> inline module_t create<cuda_cpp>(
	const context_t&                            context,
	const rtc::compilation_output_t<cuda_cpp>&  compiled_program,
	const link::options_t&                      options)
{
	if (not compiled_program.succeeded()) {
		throw ::std::invalid_argument("Attempt to create a module after compilation failure of "
			+ cuda::rtc::program::detail_::identify<cuda_cpp>(compiled_program.program_handle()));
	}
#if CUDA_VERSION >= 11030
	auto cubin = compiled_program.cubin();
	return module::create(context, cubin, options);
#else
	// Note this is less likely to succeed :-(
	auto ptx = compiled_program.ptx();
	return module::create(context, ptx, options);
#endif
}

#if CUDA_VERSION >= 11010
template<> inline module_t create<source_kind_t::ptx>(
	const context_t&                                      context,
	const rtc::compilation_output_t<source_kind_t::ptx>&  compiled_program,
	const link::options_t&                                options)
{
	if (not compiled_program.succeeded()) {
		throw ::std::invalid_argument("Attempt to create a module after compilation failure of "
			+ cuda::rtc::program::detail_::identify<source_kind_t::ptx>(compiled_program.program_handle()));
	}
	auto cubin = compiled_program.cubin();
	return module::create(context, cubin, options);
}
#endif // CUDA_VERSION >= 11010


template <source_kind_t Kind>
inline module_t create(
	device_t&                               device,
	const rtc::compilation_output_t<Kind>&  compiled_program,
	const link::options_t&                  options)
{
	return create(device.primary_context(), compiled_program, options);
}

} // namespace module

} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVRTC_OUTPUT_HPP_
