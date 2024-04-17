/**
 * @file
 *
 * @brief Contains the @ref cuda::rtc::compilation_output_t class and related code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_RTC_OUTPUT_HPP_
#define CUDA_API_WRAPPERS_RTC_OUTPUT_HPP_

#include "compilation_options.hpp"
#include "error.hpp"
#include "types.hpp"
#include "../api.hpp"

#include <vector>
#include <iostream>

namespace cuda {

///@cond
class device_t;
class context_t;
///@endcond

namespace rtc {

/**
 * The output produced by a compilation process by one of the CUDA libraries,
 * including any byproducts.
 *
 * @tparam Kind Which language was compiled to produce the result
 *
 * @note A failed compilation is also a (useful) kind of compilation output.
 */
template <source_kind_t Kind>
class compilation_output_t;

} // namespace rtc

///@cond
namespace link {
struct options_t;
} // namespace link

namespace device {
class primary_context_t;
} // namespace device

class module_t;
///@endcond

namespace module {

/// Build a contextualized module from the results of a successful compilation
template <source_kind_t Kind>
inline module_t create(
	const context_t&                        context,
	const rtc::compilation_output_t<Kind>&  compilation_output,
	const link::options_t&                  options = {});

} // namespace module

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
inline size_t get_log_size(program::handle_t<Kind> program_handle, const char* program_name);

template <>
inline size_t get_log_size<cuda_cpp>(program::handle_t<cuda_cpp> program_handle, const char* program_name)
{
	size_t size;
	auto status = nvrtcGetProgramLogSize(program_handle, &size);
	throw_if_error<cuda_cpp>(status, "Failed obtaining compilation log size for "
		+ identify<cuda_cpp>(program_handle, program_name));
	return (size > 0) ? size - 1 : 0;
}

#if CUDA_VERSION >= 11010
template <>
inline size_t get_log_size<ptx>(program::handle_t<ptx> program_handle, const char* program_name)
{
	size_t size;
	auto status = nvPTXCompilerGetErrorLogSize(program_handle, &size);
	throw_if_error<ptx>(status, "Failed obtaining compilation log size for "
		+ identify<ptx>(program_handle, program_name));
	return size;
}
#endif // CUDA_VERSION >= 11010

template <source_kind_t Kind>
inline void get_log(char* buffer, program::handle_t<Kind> program_handle, const char *program_name = nullptr);

#if CUDA_VERSION >= 11010
template <>
inline void get_log<ptx>(char* buffer, program::handle_t<ptx> program_handle, const char *program_name)
{
	auto status = nvPTXCompilerGetErrorLog(program_handle, buffer);
//			(status_t<Kind>) nvrtcGetProgramLog((handle_t<cuda_cpp>)program_handle, buffer);
	throw_if_error<ptx>(status, "Failed obtaining compilation log for "
		+ identify<ptx>(program_handle, program_name));
}
#endif

template <>
inline void get_log<cuda_cpp>(char* buffer, program::handle_t<cuda_cpp> program_handle, const char *program_name)
{
	auto status = nvrtcGetProgramLog(program_handle, buffer);
	throw_if_error<cuda_cpp>(status, "Failed obtaining compilation log for "
		+ identify<cuda_cpp>(program_handle, program_name));
}

#if CUDA_VERSION >= 11010
template <source_kind_t Kind>
inline size_t get_cubin_size_or_zero(program::handle_t<Kind> program_handle, const char* program_name);

template <>
inline size_t get_cubin_size_or_zero<ptx>(program::handle_t<ptx> program_handle, const char* program_name)
{
	size_t size;
	auto status = nvPTXCompilerGetCompiledProgramSize(program_handle, &size);
	throw_if_error<ptx>(status, "Failed obtaining program output CUBIN size for "
		+ identify<ptx>(program_handle, program_name));
	return size;
}

template <>
inline size_t get_cubin_size_or_zero<cuda_cpp>(program::handle_t<cuda_cpp> program_handle, const char* program_name)
{
	size_t size;
	auto status = nvrtcGetCUBINSize(program_handle, &size);
	throw_if_error<cuda_cpp>(status, "Failed obtaining program output CUBIN size for "
		+ identify<cuda_cpp>(program_handle, program_name));
	return size;
}

template <source_kind_t Kind, bool FailOnMissingCubin = true>
inline size_t get_cubin_size(program::handle_t<Kind> program_handle, const char* program_name)
{
	auto size = get_cubin_size_or_zero<Kind>(program_handle, program_name);
	if (FailOnMissingCubin and size == 0) {
		throw  (Kind == cuda_cpp) ?
			::std::runtime_error("Output CUBIN requested for a compilation for a virtual architecture only of "
				+ identify<Kind>(program_handle, program_name)):
			::std::runtime_error("Empty output CUBIN for compilation of "
				+ identify<Kind>(program_handle, program_name));
	}
	return size;
}

template <source_kind_t Kind>
inline void get_cubin(char* buffer, program::handle_t<Kind> program_handle, const char *program_name = nullptr);

template <>
inline void get_cubin<ptx>(char* buffer, program::handle_t<ptx> program_handle, const char *program_name)
{
	auto status = nvPTXCompilerGetCompiledProgram(program_handle, buffer);
	throw_if_error<ptx>(status, "Failed obtaining compilation output CUBIN for "
		+ identify<ptx>(program_handle, program_name));
}

template <>
inline void get_cubin<cuda_cpp>(char* buffer, program::handle_t<cuda_cpp> program_handle, const char *program_name)
{
	auto status = nvrtcGetCUBIN(program_handle, buffer);
	throw_if_error<cuda_cpp>(status, "Failed obtaining compilation output CUBIN for "
		+ identify<cuda_cpp>(program_handle, program_name));
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

inline size_t get_lto_ir_size(program::handle_t<cuda_cpp> program_handle, const char *program_name = nullptr)
{
	size_t size;
#if CUDA_VERSION >= 12000
	auto status = nvrtcGetLTOIRSize(program_handle, &size);
#else
	auto status = nvrtcGetNVVMSize(program_handle, &size);
#endif
	throw_if_rtc_error_lazy(cuda_cpp, status, "Failed obtaining output LTO IR size for compilation of "
		+ identify<cuda_cpp>(program_handle, program_name));
	return size;
}

inline void get_lto_ir(char* buffer, program::handle_t<cuda_cpp> program_handle, const char *program_name = nullptr)
{
#if CUDA_VERSION >= 12000
	auto status = nvrtcGetLTOIR(program_handle, buffer);
#else
	auto status = nvrtcGetNVVM(program_handle, buffer);
#endif
	throw_if_rtc_error_lazy(cuda_cpp, status, "Failed obtaining output LTO IR code for compilation of "
		+ identify<cuda_cpp>(program_handle, program_name));
}
#endif // CUDA_VERSION >= 11040

template <source_kind_t Kind>
status_t<Kind> destroy_and_return_status(handle_t<Kind> handle);

#if CUDA_VERSION >= 11010
template <> inline status_t<ptx> destroy_and_return_status<ptx>(handle_t<ptx> handle)
{
	return nvPTXCompilerDestroy(&handle);
}
#endif
template <> inline status_t<cuda_cpp> destroy_and_return_status<cuda_cpp>(handle_t<cuda_cpp> handle)
{
	return nvrtcDestroyProgram(&handle);
}

} // namespace detail_

} // namespace program

/// Definitions relating to and supporting the @ref compilation_output_t class
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
 * The result of the compilation of an {@ref rtc::program_t}, whether successful or
 * failed, with any related byproducts.
 *
 * @note This class _may_ own a low-level program handle.
 *
 * @note If compilation failed due to apriori-invalid arguments - an exception will
 * have been thrown. A failure indication in this class indicates a program whose
 * compilation actually _took place_ and ended with a failure.
 */
template <source_kind_t Kind>
class compilation_output_base_t {
public: // types and constants
	constexpr static const source_kind_t source_kind { Kind };
	using handle_type = program::handle_t<source_kind>;
	using status_type = status_t<source_kind>;

public: // getters

	/// @returns `true` if the compilation resulting in this output had succeeded
	bool succeeded() const { return succeeded_; }

	/// @returns `true` if the compilation resulting in this output had failed
	bool failed() const { return not succeeded_; }

	/// @returns `true` if the compilation resulting in this output had succeeded, `false` otherwise
	operator bool() const { return succeeded_; }
	const ::std::string& program_name() const { return program_name_; }
	handle_type program_handle() const { return program_handle_; }

public: // non-mutators

	/**
	 * Write a copy of the program compilation log into a user-provided buffer
	 *
     * @param[inout] buffer A writable buffer large enough to contain the compilation log
     *
	 * @return the buffer passed in (which has now been overwritten with the log)
	 *
	 * @note This will fail if the program has never been compiled, or if the
	 * buffer is not large enough to hold the complete log (plus nul character).
	 */
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

	/**
	 * Obtain a copy of the compilation log
	 *
	 * @returns an owning container with a nul-terminated copy of the log
	 *
	 * @note This will fail if the program has never been compiled.
	 */
	unique_span<char> log() const
	{
		size_t size = program::detail_::get_log_size<source_kind>(program_handle_, program_name_.c_str());
		auto result = make_unique_span<char>(size+1); // Let's append a trailing nul character, to be on the safe side
		if (size == 0) {
			result[size] = '\0';
			return result;
		}
		program::detail_::get_log<source_kind>(result.data(), program_handle_, program_name_.c_str());
		result[size] = '\0';
		return result;
	}

#if CUDA_VERSION >= 11010
	/**
	 * Write the CUBIN result of the last compilation into a buffer.
	 *
	 * @param[inout] buffer A writable buffer large enough to contain the compiled
	 *     program's CUBIN code.
	 * @return The sub-buffer, starting at the beginning of @p buffer, containing
	 *     exactly the compiled program's CUBIN (i.e. sized down to fit the contents)
	 *
	 * @note This will fail if the program has never been compiled; due to
	 * compilation failure and also due to LTO/linking failure.
	 */
	virtual span<char> cubin(span<char> buffer) const = 0;

	/**
	 * Obtain a copy of the CUBIN code resulting from the program compilation
	 *
	 * @returns an owning container with a copy of the CUBIN code
	 *
	 * @note This will fail if the program has never been compiled; if the compilation
	 * target was a virtual architecture (in which case only PTX is available); due to
	 * compilation failure and also due to LTO/linking failure.
	 */
	virtual unique_span<char> cubin() const = 0;

	/// @returns true if the program has been successfully compiled, with the result
	/// containing CUBIN code.
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
			auto status = program::detail_::destroy_and_return_status<Kind>(program_handle_);
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

/// Output of CUDA C++ code JIT-compilation
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
	 * Write a copy of the PTX resulting from the compilation into a user-provided buffer
	 *
	 * @param[inout] buffer A writable buffer large enough to contain the compiled
	 *     program's PTX code.
	 * @return The sub-buffer, starting at the beginning of @p buffer, containing
	 *     exactly the compiled program's PTX (i.e. sized down to fit the contents)
	 *
	 * @note This will throw if the program has never been compiled, or if the buffer
	 * is not large enough to contain the compiled PTX code.
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

	/**
	 * Obtain a copy of the PTX resulting from the program compilation
	 *
	 * @returns an owning container with a nul-terminated copy of the PTX code
	 *
	 * @note This will fail if the program compilation has not produced any PTX
	 */
	unique_span<char> ptx() const
	{
		size_t size = program::detail_::get_ptx_size(program_handle_, program_name_.c_str());
		auto result = make_unique_span<char>(size+1);  // Let's append a trailing nul character, to be on the safe side
		if (size == 0) {
			result[size] = '\0';
			return result;
		}
		program::detail_::get_ptx(result.data(), program_handle_, program_name_.c_str());
		result[size] = '\0';
		return result;
	}

	/// @returns true if the program has been successfully compiled, with the result containing
	/// PTX code
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
	span<char> cubin(span<char> buffer) const override
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

	unique_span<char> cubin() const override
	{
		size_t size = program::detail_::get_cubin_size<source_kind>(program_handle_, program_name_.c_str());
		auto result = make_unique_span<char>(size);
		if (size == 0) { return result; }
		program::detail_::get_cubin<source_kind>(result.data(), program_handle_, program_name_.c_str());
		return result;
	}

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
	 * Write the LTO IR result of the last compilation - the intermediate
	 * representation used for link-time optimization - into a buffer
	 *
	 * @throws ::std::invalid_argument if the supplied buffer is too small to hold
	 * the program's LTO IR.
	 *
	 * @param[inout] buffer A writable buffer large enough to contain the compiled
	 *     program's LTO IR code.
	 * @return The sub-buffer, starting at the beginning of @p buffer, containing
	 *     exactly the compiled program's LTO-IR (i.e. sized down to fit the contents)
	 *
	 * @note LTO IR was called NVVM in CUDA 11.x .
	 */
	span<char> lto_ir(span<char> buffer) const
	{
		size_t size = program::detail_::get_lto_ir_size(program_handle_, program_name_.c_str());
		if (buffer.size() < size) {
			throw ::std::invalid_argument("Provided buffer size is insufficient for the compiled program's LTO IR ("
				+ ::std::to_string(buffer.size()) + " < " + ::std::to_string(size) + ": "
				+ compilation_output::detail_::identify(*this));
		}
		program::detail_::get_lto_ir(buffer.data(), program_handle_, program_name_.c_str());
		return { buffer.data(), size };
	}

	/**
	 * Obtain a copy of the intermediate representation, for LTO purposes (LTO IR) resulting
	 * from the program compilation
	 *
	 * @returns an owning container with a nul-terminated copy of the LTO-IR code
	 *
	 * @note This will fail if the program was not compiled successfully with the LTO IR option
	 * enabled
	 */
	unique_span<char> lto_ir() const
	{
		size_t size = program::detail_::get_lto_ir_size(program_handle_, program_name_.c_str());
		auto result = make_unique_span<char>(size+1); // Let's append a trailing nul character, to be on the safe side
		if (size == 0) {
			result[size] = '\0';
			return result;
		}
		program::detail_::get_lto_ir(result.data(), program_handle_, program_name_.c_str());
		result[size] = '\0';
		return result;
	}

	/// @returns true if the program has been successfully compiled, with the result containing
	/// IR (intermediate representation) code usable for LTO (link-time optimization)
	bool has_lto_ir() const
	{
		size_t size;
#if CUDA_VERSION >= 12000
		auto status = nvrtcGetLTOIRSize(program_handle_, &size);
#else
		auto status = nvrtcGetNVVMSize(program_handle_, &size);
#endif
		if (status == NVRTC_ERROR_INVALID_PROGRAM) { return false; }
		throw_if_rtc_error_lazy(cuda_cpp, status, "Failed determining whether the NVRTC program has a compiled LTO IR result: "
			+ compilation_output::detail_::identify(*this));
		if (size == 0) {
			throw ::std::logic_error("LTO IR size reported as 0 by NVRTC for program: "
				+ compilation_output::detail_::identify(*this));
		}
		return true;
	}
#endif

	/**
	 * Obtain the mangled/lowered form of an expression registered earlier, after
	 * the compilation.
	 *
	 * @param unmangled_name A name of a `__global__` or `__device__` function or variable.
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

	/// @copydoc get_mangling_of(const char*) const
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
	span<char> cubin(span<char> buffer) const override
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

	unique_span<char> cubin() const override
	{
		size_t size = program::detail_::get_cubin_size<source_kind>(program_handle_, program_name_.c_str());
		auto result = make_unique_span<char>(size+1); // Let's append a trailing nul character, to be on the safe side
		if (size == 0) {
			result[size] = '\0';
 			return result;
		}
		program::detail_::get_cubin<source_kind>(result.data(), program_handle_, program_name_.c_str());
		result[size] = '\0';
		return result;
	}

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
inline ::std::string identify(const compilation_output_t<Kind> &compilation_output)
{
	return "Compilation output of " + program::detail_::identify<Kind>(
		compilation_output.program_handle(),
		compilation_output.program_name().c_str());
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

template<> inline module_t create<cuda_cpp>(
	const context_t&                            context,
	const rtc::compilation_output_t<cuda_cpp>&  compilation_output,
	const link::options_t&                      options)
{
	if (not compilation_output.succeeded()) {
		throw ::std::invalid_argument("Attempt to create a module after compilation failure of "
			+ cuda::rtc::program::detail_::identify<cuda_cpp>(compilation_output.program_handle()));
	}
#if CUDA_VERSION >= 11010
	auto program_handle = compilation_output.program_handle();
	auto program_name = compilation_output.program_name().c_str();
	static const bool dont_fail_on_missing_cubin { false };
	auto cubin_size = rtc::program::detail_::get_cubin_size<cuda_cpp, dont_fail_on_missing_cubin>(program_handle, program_name);
	// Note: The above won't fail even if no CUBIN was produced
	bool has_cubin = (cubin_size > 0);
	if (has_cubin) {
		auto cubin = make_unique_span<char>(cubin_size);
		rtc::program::detail_::get_cubin<cuda_cpp>(cubin.data(), program_handle, program_name);
		return module::create(context, cubin.get(), options);
	}
	// Note: At this point, we must have PTX in the output, as otherwise the compilation could
	// not have succeeded
#endif
	auto ptx = compilation_output.ptx();
	return module::create(context, ptx.get(), options);
}

#if CUDA_VERSION >= 11010
template<> inline module_t create<source_kind_t::ptx>(
	const context_t&                                      context,
	const rtc::compilation_output_t<source_kind_t::ptx>&  compilation_output,
	const link::options_t&                                options)
{
	if (not compilation_output.succeeded()) {
		throw ::std::invalid_argument("Attempt to create a module after compilation failure of "
			+ cuda::rtc::program::detail_::identify<source_kind_t::ptx>(compilation_output.program_handle()));
	}
	auto cubin = compilation_output.cubin();
	return module::create(context, cubin.get(), options);
}
#endif // CUDA_VERSION >= 11010


/// Build a module from the results of a successful compilation, in the primary context
/// of the specified device
template <source_kind_t Kind>
inline module_t create(
	device_t&                               device,
	const rtc::compilation_output_t<Kind>&  compilation_output,
	const link::options_t&                  options = {})
{
	return create(device.primary_context(), compilation_output, options);
}

} // namespace module

} // namespace cuda

#endif // CUDA_API_WRAPPERS_RTC_OUTPUT_HPP_
