/**
 * @file
 *
 * @brief Facilities for handling errors originating in NVML,
 * the NVIDIA management library, in an exception-based fashion
 * similar to that of the Driver and Runtime API wrappers.
 * Includes a basic exception class for NVML errors, wrapping
 * `::std::runtime_error`.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NVML_ERROR_HPP_
#define CUDA_API_WRAPPERS_NVML_ERROR_HPP_

#include "types.hpp"

#include <cuda_runtime_api.h>

#include <type_traits>
#include <string>
#include <stdexcept>

namespace cuda {

namespace nvml {

namespace status {

/**
 * @brief Aliases for NVNVML / PTX compilation library status codes
 */
enum named_t : ::std::underlying_type<status_t>::type {
    success = NVML_SUCCESS,
    library_not_yet_initialized                   = NVML_ERROR_UNINITIALIZED, // ... with nvml::init()
    invalid_argument                              = NVML_ERROR_INVALID_ARGUMENT,
    not_supported_on_device                       = NVML_ERROR_NOT_SUPPORTED,
    user_not_permitted                            = NVML_ERROR_NO_PERMISSION,
    // deprecated as of 11.2 at the latest - repeated initialization is allowed
    library_already_initialized                   = NVML_ERROR_ALREADY_INITIALIZED,
    queried_object_not_found                      = NVML_ERROR_NOT_FOUND,
    input_argument_not_large_enough               = NVML_ERROR_INSUFFICIENT_SIZE,
    device_power_cable_not_properly_attached      = NVML_ERROR_INSUFFICIENT_POWER,
    nvidia_driver_not_loaded                      = NVML_ERROR_DRIVER_NOT_LOADED,
    user_provided_timeout_passed                  = NVML_ERROR_TIMEOUT,
    nvidia_driver_detected_irq_issue_with_device  = NVML_ERROR_IRQ_ISSUE,
    library_couldnt_be_found_or_loaded            = NVML_ERROR_LIBRARY_NOT_FOUND,
    function_not_implemented_in_this_nvml_version = NVML_ERROR_FUNCTION_NOT_FOUND,
    device_inforom_is_corrupted                   = NVML_ERROR_CORRUPTED_INFOROM,
    gpu_device_inaccessible_on_the_bus            = NVML_ERROR_GPU_IS_LOST,
    gpu_reset_required                            = NVML_ERROR_RESET_REQUIRED,
    gpu_control_device_blocked_by_os_or_cgroups   = NVML_ERROR_OPERATING_SYSTEM,
    rm_detected_driver_library_version_mismatch   = NVML_ERROR_LIB_RM_VERSION_MISMATCH,
    operation_refused_on_busy_device              = NVML_ERROR_IN_USE,
    insufficient_memory                           = NVML_ERROR_MEMORY,
    no_data                                       = NVML_ERROR_NO_DATA,
    requested_vgpu_operation_unsupported_because_ecc_unsupported
                                                  = NVML_ERROR_VGPU_ECC_NOT_SUPPORTED,
    insufficient_non_memory_resources             = NVML_ERROR_INSUFFICIENT_RESOURCES,
    unknown_internal_error                        = NVML_ERROR_UNKNOWN
};

///@cond
constexpr bool operator==(const status_t& lhs, const named_t&  rhs) { return lhs == static_cast<status_t>(rhs); }
constexpr bool operator!=(const status_t& lhs, const named_t&  rhs) { return lhs != static_cast<status_t>(rhs); }
constexpr bool operator==(const named_t&  lhs, const status_t& rhs) { return static_cast<status_t>(lhs) == rhs; }
constexpr bool operator!=(const named_t&  lhs, const status_t& rhs) { return static_cast<status_t>(lhs) != rhs; }
///@endcond

} // namespace status

} // namespace nvml


/**
 * @brief Determine whether the API call returning the specified status had succeeded
 */
 ///@{
constexpr bool is_success(nvml::status_t status)
{
	return (status == static_cast<nvml::status_t>(nvml::status::named_t::success));
}
///@}

/**
 * @brief Determine whether the API call returning the specified status had failed
 */
constexpr bool is_failure(nvml::status_t status)
{
	return not is_success(status);
}

/**
 * Obtain a brief textual explanation for a specified NVML API return status / error code.
 *
 * @todo Consider returning a string view
 */
inline ::std::string describe(nvml::status_t status)
{
    const char *result = nvmlErrorString(status);
    if (not result) { return "Unknown error"; }
    return ::std::string{result};
}

/*

	using named = nvml::status::named_t;
	switch(status) {
	case named::success: break;
    case named:: : return "";
    case named:: : return "";
    case named:: : return "";
    case named:: : return "";
    case named:: : return "";
    case named:: : return "";
    case named:: : return "";
    case named:: : return "";
    case named:: : return "";
    case named:: : return "";
    case named:: : return "";
    case named::insufficient_memory : return "Insufficient memory";
    case named::no_data : return "No data";
    case named::requested_vgpu_operation_unsupported_because_ecc_unsupported : return "The requested vgpu operation is not available on target device, becasue ECC is enabled";
    case named::insufficient_non_memory_resources : return "Ran out of critical resources, other than memory";
	case named::unknown_internal_error : return "An internal driver error occurred";
#if CUDA_VERSION >= 12010
	case named::unsupported_device_side_sync: return "Unsupported device-side synchronization";
#endif
	}
	return "unknown error";
#endif // CUDA_VERSION >= 11010
 */

namespace nvml {

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
	runtime_error(status_t error_code, ::std::string what_arg) :
		::std::runtime_error(::std::move(what_arg) + ": " + describe(error_code)),
		code_(error_code)
	{ }
	///@endcond
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

} // namespace nvml

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
inline void throw_if_error(nvml::status_t status, const ::std::string& message) noexcept(false)
{
	if (is_failure(status)) { throw nvml::runtime_error(status, message); }
}

/**
 * Does nothing - unless the status indicates an error, in which case
 * a @ref cuda::runtime_error exception is thrown
 *
 * @param status should be @ref cuda::status::success - otherwise an exception is thrown
 */
inline void throw_if_error(nvml::status_t status) noexcept(false)
{
	if (is_failure(status)) { throw nvml::runtime_error(status); }
}

#define throw_if_nvml_error_lazy(status__, ... ) \
do { \
	::cuda::nvml::status_t tie_status__ = static_cast<::cuda::nvml::status_t>(status__); \
	if (::cuda::is_failure(tie_status__)) { \
		throw ::cuda::nvml::runtime_error(tie_status__, (__VA_ARGS__)); \
	} \
} while(false)

} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVML_ERROR_HPP_
