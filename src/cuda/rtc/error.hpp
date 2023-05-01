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

#include "types.hpp"

#include <cuda_runtime_api.h>

#include <type_traits>
#include <string>
#include <stdexcept>

namespace cuda {

namespace rtc {

namespace detail_ {

// We would _like_ to define the named status codes here. Unfortunately - we cannot, due to a
// C++11 corner-case behavior issues in GCC and/or clang. See:
// https://stackoverflow.com/q/73479613/1593077
// https://cplusplus.github.io/CWG/issues/1485
//
// template <> enum types<cuda_cpp>::named_status ...
// template <> enum types<ptx>::named_status ...

} // namespace detail_

namespace status {

/**
 * @brief Aliases for NVRTC / PTX compilation library status codes
 */
template <source_kind_t Kind>
using named_t = typename rtc::detail_::types<Kind>::named_status;

///@cond
template <source_kind_t Kind>
constexpr bool operator==(const status_t<Kind>& lhs, const named_t<Kind>& rhs) { return lhs == (status_t<Kind>) rhs;}
template <source_kind_t Kind>
constexpr bool operator!=(const status_t<Kind>& lhs, const named_t<Kind>& rhs) { return lhs != (status_t<Kind>) rhs;}
template <source_kind_t Kind>
constexpr bool operator==(const named_t<Kind>& lhs, const status_t<Kind>& rhs) { return (status_t<Kind>) lhs == rhs;}
template <source_kind_t Kind>
constexpr bool operator!=(const named_t<Kind>& lhs, const status_t<Kind>& rhs) { return (status_t<Kind>) lhs != rhs;}
///@endcond

} // namespace status

} // namespace rtc


/**
 * @brief Determine whether the API call returning the specified status had succeeded
 */
 ///@{
 template <source_kind_t Kind>
constexpr bool is_success(rtc::status_t<Kind> status)
{
	return (status == (rtc::status_t<Kind>) rtc::status::named_t<Kind>::success);
}
///@}

/**
 * @brief Determine whether the API call returning the specified status had failed
 */
template <source_kind_t Kind>
constexpr bool is_failure(rtc::status_t<Kind> status)
{
	return not is_success<Kind>(status);
}

/**
 * Obtain a brief textual explanation for a specified kind of CUDA Runtime API status
 * or error code.
 */
///@{
inline ::std::string describe(rtc::status_t<cuda_cpp> status)
{
	return nvrtcGetErrorString(status);
}

#if CUDA_VERSION >= 11010
inline ::std::string describe(rtc::status_t<ptx> status)
{
	using named = rtc::status::named_t<ptx>;
	switch(status) {
	case named::success: break;
	case named::invalid_program_handle: return "Invalid PTX compilation handle";
	case named::out_of_memory: return "out of memory";
	case named::invalid_input: return "Invalid input for PTX compilation";
	case named::compilation_invocation_incomplete: return "PTX compilation invocation incomplete";
	case named::compilation_failure: return "PTX compilation failure";
	case named::unsupported_ptx_version: return "Unsupported PTX version";
	case named::internal_error: return "Unknown PTX compilation error";
#if CUDA_VERSION >= 12010
	case named::unsupported_device_side_sync: return "Unsupported device-side synchronization";
#endif
	}
	return "unknown error";
}
#endif // CUDA_VERSION >= 11010
///@}

namespace rtc {

/**
 * A (base?) class for exceptions raised by CUDA code; these errors are thrown by
 * essentially all CUDA Runtime API wrappers upon failure.
 *
 * A CUDA runtime error can be constructed with either just a CUDA error code
 * (=status code), or a code plus an additional message.
 */
template <source_kind_t Kind>
class runtime_error : public ::std::runtime_error {
public:
	///@cond
	// TODO: Constructor chaining; and perhaps allow for more construction mechanisms?
	runtime_error(status_t<Kind> error_code) :
		::std::runtime_error(describe(error_code)),
		code_(error_code)
	{ }
	// I wonder if I should do this the other way around
	runtime_error(status_t<Kind> error_code, const ::std::string& what_arg) :
		::std::runtime_error(what_arg + ": " + describe(error_code)),
		code_(error_code)
	{ }
	///@endcond
	runtime_error(status::named_t<Kind> error_code) :
		runtime_error(static_cast<status_t<Kind>>(error_code)) { }
	runtime_error(status::named_t<Kind> error_code, const ::std::string& what_arg) :
		runtime_error(static_cast<status_t<Kind>>(error_code), what_arg) { }

	/**
	 * Obtain the CUDA status code which resulted in this error being thrown.
	 */
	status_t<Kind> code() const { return code_; }

private:
	status_t<Kind> code_;
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
template <source_kind_t Kind>
inline void throw_if_error(rtc::status_t<Kind> status, const ::std::string& message) noexcept(false)
{
	if (is_failure<Kind>(status)) { throw rtc::runtime_error<Kind>(status, message); }
}

/**
 * Does nothing - unless the status indicates an error, in which case
 * a @ref cuda::runtime_error exception is thrown
 *
 * @param status should be @ref cuda::status::success - otherwise an exception is thrown
 */
template <source_kind_t Kind>
inline void throw_if_error(rtc::status_t<Kind> status) noexcept(false)
{
	if (is_failure(status)) { throw rtc::runtime_error<Kind>(status); }
}

#define throw_if_rtc_error_lazy(Kind, status__, ... ) \
do { \
	rtc::status_t<Kind> tie_status__ = static_cast<rtc::status_t<Kind>>(status__); \
	if (is_failure<Kind>(tie_status__)) { \
		throw rtc::runtime_error<Kind>(tie_status__, (__VA_ARGS__)); \
	} \
} while(false)

} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVRTC_ERROR_HPP_
