/**
 * @file error.hpp
 *
 * @brief facilities for exception-based handling of Runtime API
 * errors, including a basic exception class wrapping
 * {@code std::runtime_error}.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_ERROR_HPP_
#define CUDA_API_WRAPPERS_ERROR_HPP_

#include "cuda/api/types.h"

#ifdef HAVE_KERNEL_TESTER_UTILS
// This wrapper exception will include a stack trace;
// also, it can take multiple arguments in the ctor which will
// be piped into a stringstream; but we won't make use of that here
// to ensure compatibility with the single-string ctor of an
// std::runtime_error
#include "util/exception.h"
using cuda_inspecific_runtime_error = util::runtime_error;
#else
#include <stdexcept>
using cuda_inspecific_runtime_error = std::runtime_error;
#endif

#include <cuda_runtime_api.h>
#include <type_traits>
#include <string>

namespace cuda {

namespace error {

// We can't just 'inherit' from status_t, unfortunately,
// so we're creating an "unrelated" enum; see also the comparison
// operators below which help us avoid the warnings we would
// get from comparing values of the two enums
enum code_t : std::underlying_type<status_t>::type {
	success                         = cudaSuccess,
	missing_configuration           = cudaErrorMissingConfiguration,
	memory_allocation               = cudaErrorMemoryAllocation,
	initialization_error            = cudaErrorInitializationError,
	launch_failure                  = cudaErrorLaunchFailure,
	prior_launch_failure            = cudaErrorPriorLaunchFailure,
	launch_timeout                  = cudaErrorLaunchTimeout,
	launch_out_of_resources         = cudaErrorLaunchOutOfResources,
	invalid_device_function         = cudaErrorInvalidDeviceFunction,
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
	cudart_unloading                = cudaErrorCudartUnloading,
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

inline bool operator==(const status_t& lhs, const code_t& rhs) { return lhs == (status_t) rhs;}
inline bool operator!=(const status_t& lhs, const code_t& rhs) { return lhs != (status_t) rhs;}

} // namespace error

inline bool is_success(status_t status)  { return status == error::success; }
inline bool is_failure(status_t status)  { return status != error::success; }

namespace detail {

template <typename I, bool UpperCase = false>
std::string as_hex(I x, unsigned hex_string_length = 2*sizeof(I) )
{
	static_assert(std::is_unsigned<I>::value, "only signed representations are supported");
	enum { bits_per_hex_digit = 4 }; // = log_2 of 16
	static const char* digit_characters =
		UpperCase ? "0123456789ABCDEF" : "0123456789abcdef" ;

	std::string result(hex_string_length,'0');
	for (unsigned digit_index = 0; digit_index < hex_string_length ; digit_index++)
	{
		size_t bit_offset = (hex_string_length - 1 - digit_index) * bits_per_hex_digit;
		auto hexadecimal_digit = (x >> bit_offset) & 0xF;
		result[digit_index] = digit_characters[hexadecimal_digit];
	}
    return result;
}

// TODO: Perhaps find a way to avoid the extra function, so that as_hex() can
// be called for pointer types as well? Would be easier with boost's uint<T>...
template <typename I, bool UpperCase = false>
inline std::string ptr_as_hex(const I* ptr, unsigned hex_string_length = 2*sizeof(I*) )
{
	return as_hex((size_t) ptr, hex_string_length);
}

} // namespace detail

inline std::string interpret_status(status_t status) { return cudaGetErrorString(status); }
inline std::string interpret_error(error::code_t code) { return interpret_status((status_t) code); }

/**
 * A (base?) class for exceptions raised by CUDA code
 */
class runtime_error : public cuda_inspecific_runtime_error {
public:
	// TODO: Constructor chaining; and perhaps allow for more construction mechanisms?
	runtime_error(cuda::status_t error_code) :
		cuda_inspecific_runtime_error(interpret_status(error_code)),
		code_(error_code)
	{ }
	// I wonder if I should do this the other way around
	runtime_error(cuda::status_t error_code, const std::string& what_arg) :
		cuda_inspecific_runtime_error(what_arg + ": " + interpret_status(error_code)),
		code_(error_code)
	{ }

	status_t code() const { return code_; }

private:
	status_t code_;
};

// TODO: The following could use std::optiomal arguments - which would
// prevent the need for dual versions of the functions - but we're
// not writing C++17 here

inline void throw_if_error(
	cuda::status_t error_code, std::string message) noexcept(false)
{
	if (is_failure(error_code)) { throw runtime_error(error_code, message); }
}

inline void throw_if_error(cuda::status_t error_code) noexcept(false)
{
	if (is_failure(error_code)) { throw runtime_error(error_code); }
}

namespace errors {
enum : bool {
	dont_clear = false,
	clear = true
};
} // namespace errors

inline void ensure_no_outstanding_error(
	std::string message, bool clear_any_error = errors::clear) noexcept(false)
{
	auto last_status = clear_any_error ? cudaGetLastError() : cudaPeekAtLastError();
	throw_if_error(last_status, message);
}

inline void ensure_no_outstanding_error(bool clear_any_error = errors::clear) noexcept(false)
{
	auto last_status = clear_any_error ? cudaGetLastError() : cudaPeekAtLastError();
	throw_if_error(last_status);
}

/**
 * Reset the CUDA status to cuda::error::success.
 */
inline void clear_outstanding_errors() { cudaGetLastError(); }

} // namespace cuda

#endif /* CUDA_API_WRAPPERS_ERROR_HPP_ */
