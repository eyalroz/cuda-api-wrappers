/**
 * @file
 *
 * @brief Facilities for exception-based handling of errors originating
 * to the NVRTC library, including a basic exception class
 * wrapping `::std::runtime_error`.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NVRTC_ERROR_HPP_
#define CUDA_API_WRAPPERS_NVRTC_ERROR_HPP_

#include <cuda/rtc/types.hpp>

#include <cuda_runtime_api.h>

#include <type_traits>
#include <string>
#include <stdexcept>

namespace cuda {

namespace rtc {

using status_t = nvrtcResult;

namespace status {

/**
 * Aliases for CUDA status codes
 *
 * @note unfortunately, this enum can't inherit from @ref cuda::status_t
 */
enum named_t : ::std::underlying_type<status_t>::type {
	success = NVRTC_SUCCESS,
	out_of_memory = NVRTC_ERROR_OUT_OF_MEMORY,
	program_creation_failure = NVRTC_ERROR_OUT_OF_MEMORY,
	invalid_input = NVRTC_ERROR_PROGRAM_CREATION_FAILURE,
	invalid_program = NVRTC_ERROR_INVALID_PROGRAM,
	invalid_option = NVRTC_ERROR_INVALID_OPTION,
	compilation_failure = NVRTC_ERROR_COMPILATION,
	builtin_operation_failure = NVRTC_ERROR_BUILTIN_OPERATION_FAILURE,
	no_registered_globals_after_compilation = NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
	no_lowered_names_before_compilation = NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
	invalid_expression_to_register_as_global = NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID,
	internal_error = NVRTC_ERROR_INTERNAL_ERROR,
};

///@cond
constexpr inline bool operator==(const status_t& lhs, const named_t& rhs) { return lhs == (status_t) rhs;}
constexpr inline bool operator!=(const status_t& lhs, const named_t& rhs) { return lhs != (status_t) rhs;}
constexpr inline bool operator==(const named_t& lhs, const status_t& rhs) { return (status_t) lhs == rhs;}
constexpr inline bool operator!=(const named_t& lhs, const status_t& rhs) { return (status_t) lhs != rhs;}
///@endcond

} // namespace status

} // namespace rtc


/**
 * @brief Determine whether the API call returning the specified status had succeeded
 */
constexpr inline bool is_success(rtc::status_t status)  { return status == (rtc::status_t) rtc::status::success; }

/**
 * @brief Determine whether the API call returning the specified status had failed
 */
constexpr inline bool is_failure(rtc::status_t status)  { return status != (rtc::status_t) rtc::status::success; }

/**
 * Obtain a brief textual explanation for a specified kind of CUDA Runtime API status
 * or error code.
 */
inline ::std::string describe(rtc::status_t status) { return nvrtcGetErrorString(status); }

namespace rtc {

/**
 * A (base?) class for exceptions raised by CUDA code; these errors are thrown by
 * essentially all CUDA Runtime API wrappers upon failure.
 *
 * A CUDA runtime error can be constructed with either just a CUDA error code
 * (=status code), or a code plus an additional message.
 */
class runtime_error : public ::std::runtime_error {
public:
	///@cond
	// TODO: Constructor chaining; and perhaps allow for more construction mechanisms?
	runtime_error(status_t error_code) :
		::std::runtime_error(describe(error_code)),
		code_(error_code)
	{ }
	// I wonder if I should do this the other way around
	runtime_error(status_t error_code, const ::std::string& what_arg) :
		::std::runtime_error(what_arg + ": " + describe(error_code)),
		code_(error_code)
	{ }
	///@endcond
	runtime_error(status::named_t error_code) :
		runtime_error(static_cast<status_t>(error_code)) { }
	runtime_error(status::named_t error_code, const ::std::string& what_arg) :
		runtime_error(static_cast<status_t>(error_code), what_arg) { }

	/**
	 * Obtain the CUDA status code which resulted in this error being thrown.
	 */
	status_t code() const { return code_; }

private:
	status_t code_;
};

} // namespace rtc

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
inline void throw_if_error(rtc::status_t status, const ::std::string& message) noexcept(false)
{
	if (is_failure(status)) { throw rtc::runtime_error(status, message); }
}

/**
 * Does nothing - unless the status indicates an error, in which case
 * a @ref cuda::runtime_error exception is thrown
 *
 * @param status should be @ref cuda::status::success - otherwise an exception is thrown
 */
inline void throw_if_error(rtc::status_t status) noexcept(false)
{
	if (is_failure(status)) { throw rtc::runtime_error(status); }
}

} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVRTC_ERROR_HPP_
