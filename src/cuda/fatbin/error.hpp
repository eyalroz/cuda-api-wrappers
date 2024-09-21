/**
 * @file
 *
 * @brief Facilities for exception-based handling of errors originating
 * in NVIDIA's fatbin creating library (nvFatbin), including a basic exception
 * class wrapping `::std::runtime_error`.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_FATBIN_BUILDER_ERROR_HPP_
#define CUDA_API_WRAPPERS_FATBIN_BUILDER_ERROR_HPP_

#include "types.hpp"

#include <nvFatbin.h>

#include <type_traits>
#include <string>
#include <stdexcept>

#if CUDA_VERSION >= 12040

namespace cuda {

namespace fatbin_builder {

namespace status {

enum named_t : ::std::underlying_type<status_t>::type {
	success = NVFATBIN_SUCCESS,
	other_internal_error = NVFATBIN_ERROR_INTERNAL,
	elf_architecture_mismatch = NVFATBIN_ERROR_ELF_ARCH_MISMATCH,
	elf_architecture_size = NVFATBIN_ERROR_ELF_SIZE_MISMATCH,
	missing_ptx_version = NVFATBIN_ERROR_MISSING_PTX_VERSION,
	unexpected_null_pointer = NVFATBIN_ERROR_NULL_POINTER,
	data_compression_failed = NVFATBIN_ERROR_COMPRESSION_FAILED,
	maximum_compressed_size_exceeded = NVFATBIN_ERROR_COMPRESSED_SIZE_EXCEEDED,
	unrecognized_option = NVFATBIN_ERROR_UNRECOGNIZED_OPTION,
	invalid_architecture = NVFATBIN_ERROR_INVALID_ARCH,
	invalid_ltoir_data = NVFATBIN_ERROR_INVALID_NVVM,
	invalid_nnvm_data = NVFATBIN_ERROR_INVALID_NVVM, // Note alias to same value
	empty_input = NVFATBIN_ERROR_EMPTY_INPUT,
#if CUDA_VERSION >= 12050
	missing_ptx_architecture = NVFATBIN_ERROR_MISSING_PTX_ARCH,
	ptx_architecture_mismatch = NVFATBIN_ERROR_PTX_ARCH_MISMATCH,
	missing_fatbin = NVFATBIN_ERROR_MISSING_FATBIN,
	invalid_index = NVFATBIN_ERROR_INVALID_INDEX,
	identifier_reused = NVFATBIN_ERROR_IDENTIFIER_REUSE,
#endif
#if CUDA_VERSION >= 12060
	internal_ptx_option_related_error = NVFATBIN_ERROR_INTERNAL_PTX_OPTION
#endif
};

///@cond
constexpr bool operator==(const status_t& lhs, const named_t& rhs) { return lhs == static_cast<status_t>(rhs); }
constexpr bool operator!=(const status_t& lhs, const named_t& rhs) { return lhs != static_cast<status_t>(rhs); }
constexpr bool operator==(const named_t& lhs, const status_t& rhs) { return static_cast<status_t>(lhs) == rhs; }
constexpr bool operator!=(const named_t& lhs, const status_t& rhs) { return static_cast<status_t>(lhs) != rhs; }
///@endcond

} // namespace status

} // namespace fatbin_builder

/**
 * @brief Determine whether the API call returning the specified status had succeeded
 */
 ///@{
constexpr bool is_success(fatbin_builder::status_t status)
{
	return (status == fatbin_builder::status::named_t::success);
}
///@}

/**
 * @brief Determine whether the API call returning the specified status had failed
 */
constexpr bool is_failure(fatbin_builder::status_t status)
{
	return not is_success(status);
}

/**
 * Obtain a brief textual explanation for a specified kind of CUDA Runtime API status
 * or error code.
 */
///@{
inline ::std::string describe(fatbin_builder::status_t status)
{
	return nvFatbinGetErrorString(status);
}

namespace fatbin_builder {

/**
 * A (base?) class for exceptions raised by CUDA code; these errors are thrown by
 * essentially all CUDA Runtime API wrappers upon failure.
 *
 * A CUDA runtime error can be constructed with either just a CUDA error code
 * (=status code), or a code plus an additional message.
 */
class runtime_error : public ::std::runtime_error {
public:
	// TODO: Constructor chaining; and perhaps allow for more construction mechanisms?
	runtime_error(status_t error_code) :
		::std::runtime_error(describe(error_code)),
		code_(error_code)
	{ }
	// I wonder if I should do this the other way around
	runtime_error(status_t error_code, ::std::string what_arg) :
		::std::runtime_error(::std::move(what_arg) + ": " + describe(error_code)),
		code_(error_code)
	{ }
	runtime_error(status::named_t error_code) :
		runtime_error(static_cast<status_t>(error_code)) { }
	runtime_error(status::named_t error_code, const ::std::string& what_arg) :
		runtime_error(static_cast<status_t>(error_code), what_arg) { }

protected:
	runtime_error(status_t error_code, ::std::runtime_error err) :
		::std::runtime_error(::std::move(err)), code_(error_code)
	{ }

public:
	static runtime_error with_message_override(status_t error_code, ::std::string complete_what_arg)
	{
		return runtime_error(error_code, ::std::runtime_error(complete_what_arg));
	}

	/**
	 * Obtain the CUDA status code which resulted in this error being thrown.
	 */
	status_t code() const { return code_; }

private:
	status_t code_;
};


} // namespace fatbin_builder

// TODO: The following could use ::std::optional arguments - which would
// prevent the need for dual versions of the functions - but we're
// not writing C++17 here

/**
 * Do nothing... unless the status indicates an error, in which case
 * a @ref cuda::runtime_error exception is thrown
 *
 * @param status should be @ref cuda::status::success - otherwise an exception is thrown
 * @param message An extra description message to add to the exception
 */
inline void throw_if_error(fatbin_builder::status_t status, const ::std::string& message) noexcept(false)
{
	if (is_failure(status)) { throw fatbin_builder::runtime_error(status, message); }
}

/**
 * Does nothing - unless the status indicates an error, in which case
 * a @ref cuda::runtime_error exception is thrown
 *
 * @param status should be @ref cuda::status::success - otherwise an exception is thrown
 */
inline void throw_if_error(fatbin_builder::status_t status) noexcept(false)
{
	if (is_failure(status)) { throw fatbin_builder::runtime_error(status); }
}

/**
 * Throws a @ref ::cuda::fatbin_builder::runtime_error exception if the status is not success
 *
 * @note The rationale for this macro is that neither the exception, nor its constructor
 * arguments, are evaluated on the "happy path"; and that cannot be achieved with a
 * function - which genertally/typically evaluates its arguments. To guarantee this
 * lazy evaluation with a function, we would need exception-construction-argument-producing
 * lambdas, which we would obviously rather avoid.
 */
#define throw_if_fatbin_builder_error_lazy(Kind, status__, ... ) \
do { \
	::cuda::fatbin_builder::status_t tie_status__ = static_cast<::cuda::fatbin_builder::status_t>(status__); \
	if (::cuda::is_failure<Kind>(tie_status__)) { \
		throw ::cuda::fatbin_builder::runtime_error<Kind>(tie_status__, (__VA_ARGS__)); \
	} \
} while(false)

} // namespace cuda

#endif // CUDA_VERSION >= 12040

#endif // CUDA_API_WRAPPERS_FATBIN_BUILDER_ERROR_HPP_
