/**
 * @file
 *
 * @brief Facilities for exception-based handling of Runtime
 * and Driver API errors, including a basic exception class
 * wrapping `::std::runtime_error`.
 *
 * @note Does not - for now - support wrapping errors generated
 * by other CUDA-related libraries like NVRTC.
 *
 * @note Unlike the Runtime API, the driver API has no memory
 * of "non-sticky" errors, which do not corrupt the current
 * context.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_ERROR_HPP_
#define CUDA_API_WRAPPERS_ERROR_HPP_

#include <cuda/api/types.hpp>
#include <cuda_runtime_api.h>
#include <cuda.h>

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
	success                          = CUDA_SUCCESS,
	memory_allocation_failure        = CUDA_ERROR_OUT_OF_MEMORY, // corresponds to cudaErrorMemoryAllocation
	initialization_error             = CUDA_ERROR_NOT_INITIALIZED, // corresponds to cudaErrorInitializationError
	already_deinitialized            = CUDA_ERROR_DEINITIALIZED, // corresponds to cudaErrorCudartUnloading
	profiler_disabled                = CUDA_ERROR_PROFILER_DISABLED,
#if CUDA_VERSION >= 10100
	profiler_not_initialized         = CUDA_ERROR_PROFILER_NOT_INITIALIZED,
#endif
	profiler_already_started         = CUDA_ERROR_PROFILER_ALREADY_STARTED,
	profiler_already_stopped         = CUDA_ERROR_PROFILER_ALREADY_STOPPED,
#if CUDA_VERSION >= 11100
	stub_library                     = CUDA_ERROR_STUB_LIBRARY,
	device_not_licensed              = CUDA_ERROR_DEVICE_NOT_LICENSED,
#endif
	prior_launch_failure             = cudaErrorPriorLaunchFailure,
	launch_timeout                   = CUDA_ERROR_LAUNCH_TIMEOUT,
	launch_out_of_resources          = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
	kernel_launch_incompatible_texturing_mode = CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
	invalid_kernel_function          = cudaErrorInvalidDeviceFunction,
	invalid_configuration            = cudaErrorInvalidConfiguration,
	invalid_device                   = CUDA_ERROR_INVALID_DEVICE,
	invalid_value                    = CUDA_ERROR_INVALID_VALUE,
	invalid_pitch_value              = cudaErrorInvalidPitchValue,
	invalid_symbol                   = cudaErrorInvalidSymbol,
	map_buffer_object_failed         = CUDA_ERROR_MAP_FAILED, // corresponds to cudaErrorMapBufferObjectFailed,
	unmap_buffer_object_failed       = CUDA_ERROR_UNMAP_FAILED, // corresponds to cudaErrorUnmapBufferObjectFailed,
	array_still_mapped               = CUDA_ERROR_ARRAY_IS_MAPPED,
	resource_already_mapped          = CUDA_ERROR_ALREADY_MAPPED,
	resource_already_acquired        = CUDA_ERROR_ALREADY_ACQUIRED,
	resource_not_mapped              = CUDA_ERROR_NOT_MAPPED,
	not_mapped_as_pointer            = CUDA_ERROR_NOT_MAPPED_AS_POINTER,
	not_mapped_as_array              = CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
	invalid_host_pointer             = cudaErrorInvalidHostPointer,
	invalid_device_pointer           = cudaErrorInvalidDevicePointer,
	invalid_texture                  = cudaErrorInvalidTexture,
	invalid_texture_binding          = cudaErrorInvalidTextureBinding,
	invalid_channel_descriptor       = cudaErrorInvalidChannelDescriptor,
	invalid_memcpy_direction         = cudaErrorInvalidMemcpyDirection,
	address_of_constant              = cudaErrorAddressOfConstant,
	texture_fetch_failed             = cudaErrorTextureFetchFailed,
	texture_not_bound                = cudaErrorTextureNotBound,
	synchronization_error            = cudaErrorSynchronizationError,
	invalid_filter_setting           = cudaErrorInvalidFilterSetting,
	invalid_norm_setting             = cudaErrorInvalidNormSetting,
	mixed_device_execution           = cudaErrorMixedDeviceExecution,
	unknown                          = CUDA_ERROR_UNKNOWN,
	not_yet_implemented              = cudaErrorNotYetImplemented,
	memory_value_too_large           = cudaErrorMemoryValueTooLarge,
	invalid_resource_handle          = CUDA_ERROR_INVALID_HANDLE,
#if CUDA_VERSION >= 10000
	resource_not_in_valid_state     = CUDA_ERROR_ILLEGAL_STATE,
#endif
	async_operations_not_yet_completed = CUDA_ERROR_NOT_READY,
	insufficient_driver              = cudaErrorInsufficientDriver,
	set_on_active_process            = cudaErrorSetOnActiveProcess,
	invalid_surface                  = cudaErrorInvalidSurface,
	symbol_not_found                 = CUDA_ERROR_NOT_FOUND, // corresponds to cudaErrorSymbolNotFound
	no_device                        = CUDA_ERROR_NO_DEVICE,
	ecc_uncorrectable                = CUDA_ERROR_ECC_UNCORRECTABLE,
	shared_object_symbol_not_found   = CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
	invalid_source                   = CUDA_ERROR_INVALID_SOURCE,
	file_not_found                   = CUDA_ERROR_FILE_NOT_FOUND,
	shared_object_init_failed        = CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
	jit_compiler_not_found           = CUDA_ERROR_JIT_COMPILER_NOT_FOUND,
#if CUDA_VERSION >= 11100
	unsupported_ptx_version          = CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
#endif
#if CUDA_VERSION >= 11200
	jit_compilation_disabled         = CUDA_ERROR_JIT_COMPILATION_DISABLED,
#endif
#if CUDA_VERSION >= 11400
	unsupported_exec_affinity        = CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY,
#endif
	unsupported_limit                = CUDA_ERROR_UNSUPPORTED_LIMIT,
	duplicate_variable_name          = cudaErrorDuplicateVariableName,
	duplicate_texture_name           = cudaErrorDuplicateTextureName,
	duplicate_surface_name           = cudaErrorDuplicateSurfaceName,
	devices_unavailable              = cudaErrorDevicesUnavailable,
	invalid_kernel_image             = CUDA_ERROR_INVALID_IMAGE, // corresponds to cudaErrorInvalidKernelImage,
	no_kernel_image_for_device       = CUDA_ERROR_NO_BINARY_FOR_GPU, // corresponds to cudaErrorNoKernelImageForDevice,
	incompatible_driver_context      = cudaErrorIncompatibleDriverContext,
	missing_configuration            = cudaErrorMissingConfiguration,
	invalid_context                  = CUDA_ERROR_INVALID_CONTEXT,
	context_already_current          = CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
	context_already_in_use           = CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
	peer_access_already_enabled      = CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
	peer_access_not_enabled          = CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
	device_already_in_use            = cudaErrorDeviceAlreadyInUse,
	primary_context_already_active   = CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
	context_is_destroyed             = CUDA_ERROR_CONTEXT_IS_DESTROYED,
	primary_context_is_uninitialized = CUDA_ERROR_CONTEXT_IS_DESTROYED, // an alias!
#if CUDA_VERSION >= 10200
	device_uninitialized             = cudaErrorDeviceUninitialized,
#endif
	assert                           = CUDA_ERROR_ASSERT,
	too_many_peers                   = CUDA_ERROR_TOO_MANY_PEERS,
	host_memory_already_registered   = CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
	host_memory_not_registered       = CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
	operating_system                 = CUDA_ERROR_OPERATING_SYSTEM,
	peer_access_unsupported          = CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
	launch_max_depth_exceeded        = cudaErrorLaunchMaxDepthExceeded,
	launch_file_scoped_tex           = cudaErrorLaunchFileScopedTex,
	launch_file_scoped_surf          = cudaErrorLaunchFileScopedSurf,
	sync_depth_exceeded              = cudaErrorSyncDepthExceeded,
	launch_pending_count_exceeded    = cudaErrorLaunchPendingCountExceeded,
	invalid_device_function          = cudaErrorInvalidDeviceFunction,
	not_permitted                    = CUDA_ERROR_NOT_PERMITTED,
	not_supported                    = CUDA_ERROR_NOT_SUPPORTED,
	hardware_stack_error             = CUDA_ERROR_HARDWARE_STACK_ERROR,
	illegal_instruction              = CUDA_ERROR_ILLEGAL_INSTRUCTION,
	misaligned_address               = CUDA_ERROR_MISALIGNED_ADDRESS,
	exception_during_kernel_execution = CUDA_ERROR_LAUNCH_FAILED,
	cooperative_launch_too_large     = CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
	invalid_address_space            = CUDA_ERROR_INVALID_ADDRESS_SPACE,
	invalid_pc                       = CUDA_ERROR_INVALID_PC,
	illegal_address                  = CUDA_ERROR_ILLEGAL_ADDRESS,
	invalid_ptx                      = CUDA_ERROR_INVALID_PTX,
	invalid_graphics_context         = CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
	nvlink_uncorrectable             = CUDA_ERROR_NVLINK_UNCORRECTABLE,
	startup_failure                  = cudaErrorStartupFailure,
	api_failure_base                 = cudaErrorApiFailureBase,
#if CUDA_VERSION >= 10000
	system_not_ready                 = CUDA_ERROR_SYSTEM_NOT_READY,
#endif
#if CUDA_VERSION >= 10100
	system_driver_mismatch           = CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
	not_supported_on_device          = CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE,
#endif
#if CUDA_VERSION >= 10000
	stream_capture_unsupported       = CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
	stream_capture_invalidated       = CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
	stream_capture_merge             = CUDA_ERROR_STREAM_CAPTURE_MERGE,
	stream_capture_unmatched         = CUDA_ERROR_STREAM_CAPTURE_UNMATCHED,
	stream_capture_unjoined          = CUDA_ERROR_STREAM_CAPTURE_UNJOINED,
	stream_capture_isolation         = CUDA_ERROR_STREAM_CAPTURE_ISOLATION,
	stream_capture_disallowed_implicit_dependency = CUDA_ERROR_STREAM_CAPTURE_IMPLICIT,
	not_permitted_on_captured_event  = CUDA_ERROR_CAPTURED_EVENT,
#endif
#if CUDA_VERSION >= 10100
	stream_capture_wrong_thread      = CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD,
#endif
#if CUDA_VERSION >= 10200
	timeout_lapsed                   = CUDA_ERROR_TIMEOUT,
	graph_update_would_violate_constraints = CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE,
#endif
#if CUDA_VERSION >= 11400
	mps_connection_failed            = CUDA_ERROR_MPS_CONNECTION_FAILED,
	mps_rpc_failure                  = CUDA_ERROR_MPS_RPC_FAILURE,
	mps_server_not_ready             = CUDA_ERROR_MPS_SERVER_NOT_READY,
	mps_max_clients_reached          = CUDA_ERROR_MPS_MAX_CLIENTS_REACHED,
	mps_max_connections_reached      = CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED,
	async_error_in_external_device   = CUDA_ERROR_EXTERNAL_DEVICE,
#endif
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
constexpr inline bool is_failure(status_t status)  { return not is_success(status); }

/**
 * Obtain a brief textual explanation for a specified kind of CUDA Runtime API status
 * or error code.
 */
///@{
inline ::std::string describe(status_t status)
{
	const char* description;
	auto description_lookup_status = cuGetErrorString(status, &description);
	return (description_lookup_status != CUDA_SUCCESS) ? nullptr : description;
}
inline ::std::string describe(cudaError_t status) { return cudaGetErrorString(status); }
///@}


namespace detail_ {

template <typename I, bool UpperCase = false>
std::string as_hex(I x)
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
	return as_hex(reinterpret_cast<uintptr_t>(ptr));
}

} // namespace detail_

/**
 * A (base?) class for exceptions raised by CUDA code; these errors are thrown by
 * essentially all CUDA Runtime API wrappers upon failure.
 *
 * A CUDA runtime error can be constructed with either just a CUDA error code
 * (=status code), or a code plus an additional message.
 *
 * @todo Consider renaming this to avoid confusion with the CUDA Runtime.
 */
class runtime_error : public ::std::runtime_error {
public:
	///@cond
	// TODO: Constructor chaining; and perhaps allow for more construction mechanisms?
	runtime_error(status_t error_code) :
		::std::runtime_error(describe(error_code)), code_(error_code)
	{ }
	// I wonder if I should do this the other way around
	runtime_error(status_t error_code, const ::std::string& what_arg) :
		::std::runtime_error(what_arg + ": " + describe(error_code)),
		code_(error_code)
	{ }
	// I wonder if I should do this the other way around
	runtime_error(status_t error_code, ::std::string&& what_arg) :
		runtime_error(error_code, what_arg)
	{ }
	///@endcond
	explicit runtime_error(status::named_t error_code) :
		runtime_error(static_cast<status_t>(error_code)) { }
	runtime_error(status::named_t error_code, const ::std::string& what_arg) :
		runtime_error(static_cast<status_t>(error_code), what_arg) { }
	runtime_error(status::named_t error_code, ::std::string&& what_arg) :
		runtime_error(static_cast<status_t>(error_code), what_arg) { }

	/**
	 * Obtain the CUDA status code which resulted in this error being thrown.
	 */
	status_t code() const { return code_; }

private:
	status_t code_;
};

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
inline void throw_if_error(status_t status, const ::std::string& message) noexcept(false)
{
	if (is_failure(status)) { throw runtime_error(status, message); }
}

inline void throw_if_error(cudaError_t status, const ::std::string& message) noexcept(false)
{
	throw_if_error(static_cast<status_t>(status), message);
}

inline void throw_if_error(status_t status, ::std::string&& message) noexcept(false)
{
	if (is_failure(status)) { throw runtime_error(status, message); }
}

inline void throw_if_error(cudaError_t status, ::std::string&& message) noexcept(false)
{
	return throw_if_error(static_cast<status_t>(status), message);
}

/**
 * Does nothing - unless the status indicates an error, in which case
 * a @ref cuda::runtime_error exception is thrown
 *
 * @param status should be @ref cuda::status::success - otherwise an exception is thrown
 */
inline void throw_if_error(status_t status) noexcept(false)
{
	if (is_failure(status)) { throw runtime_error(status); }
}

inline void throw_if_error(cudaError_t status) noexcept(false)
{
	throw_if_error(static_cast<status_t>(status));
}

enum : bool {
	dont_clear_errors = false,
	do_clear_errors    = true
};

namespace detail_ {

namespace outstanding_runtime_error {

/**
 * Clears the current CUDA context's status and return any outstanding error.
 *
 * @todo Reconsider what this does w.r.t. driver calls
 */
inline status_t clear() noexcept
{
	return static_cast<status_t>(cudaGetLastError());
}

/**
 * Get the code of the last error in a CUDA-related action.
 *
 * @todo Reconsider what this does w.r.t. driver calls
 */
inline status_t get() noexcept
{
	return static_cast<status_t>(cudaPeekAtLastError());
}

} // namespace outstanding_runtime_error
} // namespace detail_

/**
 * Unlike the Runtime API, where every error is outstanding
 * until cleared, the Driver API, which we use mostly, only
 * remembers "sticky" errors - severe errors which corrupt
 * contexts. Such errors cannot be recovered from / cleared,
 * and require either context destruction or process termination.
 */
namespace outstanding_error {

/**
 * @return the code of a sticky (= context-corrupting) error,
 * if the CUDA driver has recently encountered any.
 */
inline status_t get()
{
	constexpr const unsigned dummy_flags{0};
	auto status = cuInit(dummy_flags);
	return static_cast<status_t>(status);
}

/**
 * @brief Does nothing (unless throwing an exception)
 *
 * @note similar to @ref cuda::throw_if_error, but uses the CUDA driver's
 * own state regarding whether or not a sticky error has occurred
 */
inline void ensure_none(const ::std::string &message) noexcept(false)
{
	auto status = get();
	throw_if_error(status, message);
}

/**
 * @brief A variant of @ref ensure_none() which takes
 * a C-style string.
 *
 * @note exists so as to avoid incorrect overload resolution of
 * `ensure_none(my_c_string)` calls.
 */
inline void ensure_none(const char *message) noexcept(false)
{
	return ensure_none(::std::string{message});
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
inline void ensure_none() noexcept(false)
{
	auto status = get();
	throw_if_error(status);
}

} // namespace outstanding_error

// The following few functions are used in the error messages
// generated for exceptions thrown by various API wrappers.

namespace device {
namespace detail_ {
inline ::std::string identify(device::id_t device_id)
{
	return ::std::string("device ") + ::std::to_string(device_id);
}
} // namespace detail_
} // namespace device

namespace context {
namespace detail_ {

inline ::std::string identify(handle_t handle)
{
	return "context " + cuda::detail_::ptr_as_hex(handle);
}

inline ::std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail_::identify(device_id);
}

} // namespace detail_

namespace current{
namespace detail_ {
inline ::std::string identify(context::handle_t handle)
{
	return "current context: " + context::detail_::identify(handle);
}
inline ::std::string identify(context::handle_t handle, device::id_t device_id)
{
	return "current context: " + context::detail_::identify(handle, device_id);
}
} // namespace detail_
} // namespace current

} // namespace context

namespace device {
namespace primary_context {
namespace detail_ {

inline ::std::string identify(handle_t handle, device::id_t device_id)
{
	return "context " + context::detail_::identify(handle, device_id);
}
inline ::std::string identify(handle_t handle)
{
	return "context " + context::detail_::identify(handle);
}
} // namespace detail_
} // namespace primary_context
} // namespace device

namespace stream {
namespace detail_ {
inline ::std::string identify(handle_t handle)
{
	return "stream " + cuda::detail_::ptr_as_hex(handle);
}
inline ::std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail_::identify(device_id);
}
inline ::std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " in " + context::detail_::identify(context_handle);
}
inline ::std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " in " + context::detail_::identify(context_handle, device_id);
}
} // namespace detail_
} // namespace stream

namespace event {
namespace detail_ {
inline ::std::string identify(handle_t handle)
{
	return "event " + cuda::detail_::ptr_as_hex(handle);
}
inline ::std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail_::identify(device_id);
}
inline ::std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " on " + context::detail_::identify(context_handle);
}
inline ::std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " on " + context::detail_::identify(context_handle, device_id);
}
} // namespace detail_
} // namespace event

namespace kernel {
namespace detail_ {

inline ::std::string identify(const void* ptr)
{
	return "kernel " + cuda::detail_::ptr_as_hex(ptr);
}
inline ::std::string identify(const void* ptr, device::id_t device_id)
{
    return identify(ptr) + " on " + device::detail_::identify(device_id);
}
inline ::std::string identify(const void* ptr, context::handle_t context_handle)
{
	return identify(ptr) + " in " + context::detail_::identify(context_handle);
}
inline ::std::string identify(const void* ptr, context::handle_t context_handle, device::id_t device_id)
{
	return identify(ptr) + " in " + context::detail_::identify(context_handle, device_id);
}
inline ::std::string identify(handle_t handle)
{
	return "kernel at " + cuda::detail_::ptr_as_hex(handle);
}
inline ::std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " in " + context::detail_::identify(context_handle);
}
inline ::std::string identify(handle_t handle,  device::id_t device_id)
{
	return identify(handle) + " on " + device::detail_::identify(device_id);
}
inline ::std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " in " + context::detail_::identify(context_handle, device_id);
}

} // namespace detail
} // namespace kernel

namespace memory {
namespace detail_ {

inline ::std::string identify(region_t region)
{
	return ::std::string("memory region at ") + cuda::detail_::ptr_as_hex(region.data())
		+ " of size " + ::std::to_string(region.size());
}

} // namespace detail_
} // namespace memory

} // namespace cuda

#endif // CUDA_API_WRAPPERS_ERROR_HPP_
