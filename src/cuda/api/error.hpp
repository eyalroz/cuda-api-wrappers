/**
 * @file error.hpp
 *
 * @brief Facilities for exception-based handling of Runtime API
 * errors, including a basic exception class wrapping
 * `::std::runtime_error`.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_ERROR_HPP_
#define CUDA_API_WRAPPERS_ERROR_HPP_

#include <cuda/common/types.hpp>

#include <cuda_runtime_api.h>
#include <type_traits>
#include <string>
#include <stdexcept>

namespace cuda {

namespace status {

/**
 * Aliases for CUDA status codes
 *
 * @note unfortunately, this enum can't inherit from @ref cuda::status_t
 */
enum named_t : ::std::underlying_type<status_t>::type {
	success                         = cudaSuccess,
	missing_configuration           = cudaErrorMissingConfiguration,
	memory_allocation               = cudaErrorMemoryAllocation,
	initialization_error            = cudaErrorInitializationError,
	launch_failure                  = cudaErrorLaunchFailure,
	prior_launch_failure            = cudaErrorPriorLaunchFailure,
	launch_timeout                  = cudaErrorLaunchTimeout,
	launch_out_of_resources         = cudaErrorLaunchOutOfResources,
	invalid_kernel_function         = cudaErrorInvalidDeviceFunction,
	invalid_configuration           = cudaErrorInvalidConfiguration,
	invalid_device                  = cudaErrorInvalidDevice,
	invalid_value                   = cudaErrorInvalidValue,
	invalid_pitch_value             = cudaErrorInvalidPitchValue,
	invalid_symbol                  = cudaErrorInvalidSymbol,
	map_buffer_object_failed        = cudaErrorMapBufferObjectFailed,
	unmap_buffer_object_failed      = cudaErrorUnmapBufferObjectFailed,
	invalid_host_pointer            = cudaErrorInvalidHostPointer,
	invalid_device_pointer          = cudaErrorInvalidDevicePointer,
	invalid_texture                 = cudaErrorInvalidTexture,
	invalid_texture_binding         = cudaErrorInvalidTextureBinding,
	invalid_channel_descriptor      = cudaErrorInvalidChannelDescriptor,
	invalid_memcpy_direction        = cudaErrorInvalidMemcpyDirection,
	address_of_constant             = cudaErrorAddressOfConstant,
	texture_fetch_failed            = cudaErrorTextureFetchFailed,
	texture_not_bound               = cudaErrorTextureNotBound,
	synchronization_error           = cudaErrorSynchronizationError,
	invalid_filter_setting          = cudaErrorInvalidFilterSetting,
	invalid_norm_setting            = cudaErrorInvalidNormSetting,
	mixed_device_execution          = cudaErrorMixedDeviceExecution,
	cuda_runtime_unloading          = cudaErrorCudartUnloading,
	unknown                         = cudaErrorUnknown,
	not_yet_implemented             = cudaErrorNotYetImplemented,
	memory_value_too_large          = cudaErrorMemoryValueTooLarge,
	invalid_resource_handle         = cudaErrorInvalidResourceHandle,
	not_ready                       = cudaErrorNotReady,
	insufficient_driver             = cudaErrorInsufficientDriver,
	set_on_active_process           = cudaErrorSetOnActiveProcess,
	invalid_surface                 = cudaErrorInvalidSurface,
	no_device                       = cudaErrorNoDevice,
	ecc_uncorrectable               = cudaErrorECCUncorrectable,
	shared_object_symbol_not_found  = cudaErrorSharedObjectSymbolNotFound,
	shared_object_init_failed       = cudaErrorSharedObjectInitFailed,
	unsupported_limit               = cudaErrorUnsupportedLimit,
	duplicate_variable_name         = cudaErrorDuplicateVariableName,
	duplicate_texture_name          = cudaErrorDuplicateTextureName,
	duplicate_surface_name          = cudaErrorDuplicateSurfaceName,
	devices_unavailable             = cudaErrorDevicesUnavailable,
	invalid_kernel_image            = cudaErrorInvalidKernelImage,
	no_kernel_image_for_device      = cudaErrorNoKernelImageForDevice,
	incompatible_driver_context     = cudaErrorIncompatibleDriverContext,
	peer_access_already_enabled     = cudaErrorPeerAccessAlreadyEnabled,
	peer_access_not_enabled         = cudaErrorPeerAccessNotEnabled,
	device_already_in_use           = cudaErrorDeviceAlreadyInUse,
	profiler_disabled               = cudaErrorProfilerDisabled,
	profiler_not_initialized        = cudaErrorProfilerNotInitialized,
	profiler_already_started        = cudaErrorProfilerAlreadyStarted,
	profiler_already_stopped        = cudaErrorProfilerAlreadyStopped,
	assert                          = cudaErrorAssert,
	too_many_peers                  = cudaErrorTooManyPeers,
	host_memory_already_registered  = cudaErrorHostMemoryAlreadyRegistered,
	host_memory_not_registered      = cudaErrorHostMemoryNotRegistered,
	operating_system                = cudaErrorOperatingSystem,
	peer_access_unsupported         = cudaErrorPeerAccessUnsupported,
	launch_max_depth_exceeded       = cudaErrorLaunchMaxDepthExceeded,
	launch_file_scoped_tex          = cudaErrorLaunchFileScopedTex,
	launch_file_scoped_surf         = cudaErrorLaunchFileScopedSurf,
	sync_depth_exceeded             = cudaErrorSyncDepthExceeded,
	launch_pending_count_exceeded   = cudaErrorLaunchPendingCountExceeded,
	not_permitted                   = cudaErrorNotPermitted,
	not_supported                   = cudaErrorNotSupported,
	hardware_stack_error            = cudaErrorHardwareStackError,
	illegal_instruction             = cudaErrorIllegalInstruction,
	misaligned_address              = cudaErrorMisalignedAddress,
	invalid_address_space           = cudaErrorInvalidAddressSpace,
	invalid_pc                      = cudaErrorInvalidPc,
	illegal_address                 = cudaErrorIllegalAddress,
	invalid_ptx                     = cudaErrorInvalidPtx,
	invalid_graphics_context        = cudaErrorInvalidGraphicsContext,
	nvlink_uncorrectable            = cudaErrorNvlinkUncorrectable,
	startup_failure                 = cudaErrorStartupFailure,
	api_failure_base                = cudaErrorApiFailureBase
};

///@cond
constexpr inline bool operator==(const status_t& lhs, const named_t& rhs) { return lhs == (status_t) rhs;}
constexpr inline bool operator!=(const status_t& lhs, const named_t& rhs) { return lhs != (status_t) rhs;}
constexpr inline bool operator==(const named_t& lhs, const status_t& rhs) { return (status_t) lhs == rhs;}
constexpr inline bool operator!=(const named_t& lhs, const status_t& rhs) { return (status_t) lhs != rhs;}
///@endcond

} // namespace status

/**
 * @brief Determine whether the API call returning the specified status had succeeded
 */
constexpr inline bool is_success(status_t status)  { return status == (status_t) status::success; }

/**
 * @brief Determine whether the API call returning the specified status had failed
 */
constexpr inline bool is_failure(status_t status)  { return status != (status_t) status::success; }

/**
 * Obtain a brief textual explanation for a specified kind of CUDA Runtime API status
 * or error code.
 */
inline ::std::string describe(status_t status) { return cudaGetErrorString(status); }

namespace detail_ {

template <typename I, bool UpperCase = false>
::std::string as_hex(I x)
{
	static_assert(::std::is_unsigned<I>::value, "only signed representations are supported");
	unsigned num_hex_digits = 2*sizeof(I);
	if (x == 0) return "0x0";

	enum { bits_per_hex_digit = 4 }; // = log_2 of 16
	static const char* digit_characters =
		UpperCase ? "0123456789ABCDEF" : "0123456789abcdef" ;

	::std::string result(num_hex_digits,'0');
	for (unsigned digit_index = 0; digit_index < num_hex_digits ; digit_index++)
	{
		size_t bit_offset = (num_hex_digits - 1 - digit_index) * bits_per_hex_digit;
		auto hexadecimal_digit = (x >> bit_offset) & 0xF;
		result[digit_index] = digit_characters[hexadecimal_digit];
	}
	return "0x0" + result.substr(result.find_first_not_of('0'), ::std::string::npos);
}

// TODO: Perhaps find a way to avoid the extra function, so that as_hex() can
// be called for pointer types as well? Would be easier with boost's uint<T>...
template <typename I, bool UpperCase = false>
inline ::std::string ptr_as_hex(const I* ptr)
{
	return as_hex((size_t) ptr);
}

} // namespace detail_

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
	runtime_error(cuda::status_t error_code) :
		::std::runtime_error(describe(error_code)),
		code_(error_code)
	{ }
	// I wonder if I should do this the other way around
	runtime_error(cuda::status_t error_code, const ::std::string& what_arg) :
		::std::runtime_error(what_arg + ": " + describe(error_code)),
		code_(error_code)
	{ }
	///@endcond
	runtime_error(cuda::status::named_t error_code) :
		runtime_error(static_cast<cuda::status_t>(error_code)) { }
	runtime_error(cuda::status::named_t error_code, const ::std::string& what_arg) :
		runtime_error(static_cast<cuda::status_t>(error_code), what_arg) { }

	/**
	 * Obtain the CUDA status code which resulted in this error being thrown.
	 */
	status_t code() const { return code_; }

private:
	status_t code_;
};

// TODO: The following could use ::std::optiomal arguments - which would
// prevent the need for dual versions of the functions - but we're
// not writing C++17 here

/**
 * Do nothing... unless the status indicates an error, in which case
 * a @ref cuda::runtime_error exception is thrown
 *
 * @param status should be @ref cuda::status::success - otherwise an exception is thrown
 * @param message An extra description message to add to the exception
 */
inline void throw_if_error(cuda::status_t status, const ::std::string& message) noexcept(false)
{
	if (is_failure(status)) { throw runtime_error(status, message); }
}

/**
 * Does nothing - unless the status indicates an error, in which case
 * a @ref cuda::runtime_error exception is thrown
 *
 * @param status should be @ref cuda::status::success - otherwise an exception is thrown
 */
inline void throw_if_error(cuda::status_t status) noexcept(false)
{
	if (is_failure(status)) { throw runtime_error(status); }
}

enum : bool {
	dont_clear_errors = false,
	do_clear_errors    = true
};

namespace outstanding_error {

/**
 * Reset the CUDA status to @ref cuda::status::success.
 */
inline status_t clear() noexcept { return cudaGetLastError();    }

/**
 * Get the code of the last error in a CUDA-related action.
 */
inline status_t get()   noexcept { return cudaPeekAtLastError(); }

/**
 * @brief Does nothing (unless throwing an exception)
 *
 * @note similar to @ref cuda::throw_if_error, but uses the CUDA Runtime API's internal
 * state
 *
 * @throws cuda::runtime_error if the CUDA runtime API has
 * encountered previously encountered an (uncleared) error
 *
 * @param message Additional message to incldue in the exception thrown
 * @param clear_any_error When true, clears the CUDA Runtime API's state from
 * recalling errors arising from before this moment
 *
 *
 */
inline void ensure_none(
	::std::string  message,
	bool         clear_any_error = do_clear_errors) noexcept(false)
{
	auto last_status = clear_any_error ? clear() : get();
	throw_if_error(last_status, message);
}

/**
 * @brief A variant of @ref ensure_none() which takes
 * a C-style string.
 *
 * @note exists so as to avoid incorrect overload resolution of
 * `ensure_none(my_c_string)` calls.
 */
inline void ensure_none(
	const char*  message,
	bool         clear_any_error = do_clear_errors) noexcept(false)
{
	return ensure_none(::std::string(message), clear_any_error);
}

/**
 * @brief Does nothing (unless throwing an exception)
 *
 * @note similar to @ref throw_if_error, but uses the CUDA Runtime API's internal
 * state
 *
 * @throws cuda::runtime_error if the CUDA runtime API has
 * encountered previously encountered an (uncleared) error
 *
 * @param clear_any_error When true, clears the CUDA Runtime API's state from
 * recalling errors arising from before this oment
 */
inline void ensure_none(bool clear_any_error = do_clear_errors) noexcept(false)
{
	auto last_status = clear_any_error ? clear() : get();
	throw_if_error(last_status);
}

} // namespace outstanding_error


} // namespace cuda

#endif // CUDA_API_WRAPPERS_ERROR_HPP_
