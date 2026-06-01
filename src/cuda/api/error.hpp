/**
 * @file
 *
 * @brief Facilities for exception-based handling of Runtime
 * and Driver API errors, including a basic exception class
 * wrapping `std::runtime_error`.
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

#include "types.hpp"
#include <cuda_runtime_api.h>

#include <type_traits>
#include <string>
#include <stdexcept>

namespace cuda_ {

namespace status {

/**
 * Aliases for CUDA status codes
 *
 * @note unfortunately, this enum can't inherit from @ref cuda_::status_t
 */
enum named_t : std::underlying_type<status_t>::type {
	// Errors defined for the CUDA driver API (including ones which are
	// duplicated in the runtime API)

	success                              = CUDA_SUCCESS, /// Operation was successful; no errors
	invalid_value                        = CUDA_ERROR_INVALID_VALUE,
	memory_allocation_failure            = CUDA_ERROR_OUT_OF_MEMORY,
	not_yet_initialized                  = CUDA_ERROR_NOT_INITIALIZED,
	already_deinitialized                = CUDA_ERROR_DEINITIALIZED,
	profiler_disabled                    = CUDA_ERROR_PROFILER_DISABLED,
#if CUDA_VERSION >= 10100
	profiler_not_initialized             = CUDA_ERROR_PROFILER_NOT_INITIALIZED,
#endif
	profiler_already_started             = CUDA_ERROR_PROFILER_ALREADY_STARTED,
	profiler_already_stopped             = CUDA_ERROR_PROFILER_ALREADY_STOPPED,
#if CUDA_VERSION >= 11100
	stub_library                         = CUDA_ERROR_STUB_LIBRARY,
#endif
#if CUDA_VERSION >= 13000
	call_requires_newer_driver           = CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER,
#endif
#if CUDA_VERSION >= 11070
	device_unavailable                   = CUDA_ERROR_DEVICE_UNAVAILABLE,
#endif
	no_device                            = CUDA_ERROR_NO_DEVICE,
	invalid_device                       = CUDA_ERROR_INVALID_DEVICE,
#if CUDA_VERSION >= 11100
	device_not_licensed                  = CUDA_ERROR_DEVICE_NOT_LICENSED,
#endif
	invalid_kernel_image                 = CUDA_ERROR_INVALID_IMAGE,
	invalid_context                      = CUDA_ERROR_INVALID_CONTEXT,
	context_already_current              = CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
	map_buffer_object_failed             = CUDA_ERROR_MAP_FAILED,
	unmap_buffer_object_failed           = CUDA_ERROR_UNMAP_FAILED,
	array_still_mapped                   = CUDA_ERROR_ARRAY_IS_MAPPED,
	resource_already_mapped              = CUDA_ERROR_ALREADY_MAPPED,
	no_binary_for_this_gpu               = CUDA_ERROR_NO_BINARY_FOR_GPU,
	resource_already_acquired            = CUDA_ERROR_ALREADY_ACQUIRED,
	resource_not_mapped                  = CUDA_ERROR_NOT_MAPPED,
	not_mapped_as_array                  = CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
	not_mapped_as_pointer                = CUDA_ERROR_NOT_MAPPED_AS_POINTER,
	ecc_uncorrectable                    = CUDA_ERROR_ECC_UNCORRECTABLE,
	unsupported_limit                    = CUDA_ERROR_UNSUPPORTED_LIMIT,
	context_already_in_use               = CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
	peer_access_unsupported              = CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
	invalid_ptx                          = CUDA_ERROR_INVALID_PTX,
	invalid_graphics_content             = CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
	nvlink_uncorrectable                 = CUDA_ERROR_NVLINK_UNCORRECTABLE,
	jit_compiler_not_found               = CUDA_ERROR_JIT_COMPILER_NOT_FOUND,
#if CUDA_VERSION >= 11100
	unsupported_ptx_version              = CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
#endif
#if CUDA_VERSION >= 11200
	jit_compilation_disabled             = CUDA_ERROR_JIT_COMPILATION_DISABLED,
#endif
#if CUDA_VERSION >= 11040
	unsupported_exec_affinity            = CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY,
#endif
	unsupported_device_sync_call_in_jit_code
	                                     = CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC,
#if CUDA_VERSION >= 12080
	opaque_error_contained_on_device     = CUDA_ERROR_CONTAINED,
#endif
	invalid_kernel_source                = CUDA_ERROR_INVALID_SOURCE,
	file_not_found                       = CUDA_ERROR_FILE_NOT_FOUND,
	shared_object_symbol_not_found       = CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
	shared_object_init_failed            = CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
	system_call_failed                   = CUDA_ERROR_OPERATING_SYSTEM,
	invalid_resource_handle              = CUDA_ERROR_INVALID_HANDLE,
#if CUDA_VERSION >= 10000
	resource_not_in_valid_state          = CUDA_ERROR_ILLEGAL_STATE,
#endif
	introspection_would_discard_info     = CUDA_ERROR_LOSSY_QUERY,
	named_symbol_not_found               = CUDA_ERROR_NOT_FOUND,
	async_dependency_ops_not_yet_completed
	                                     = CUDA_ERROR_NOT_READY,
	invalid_address_in_load_or_store     = CUDA_ERROR_ILLEGAL_ADDRESS,
	too_many_args_or_grid_threads_for_launch
	                                     = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
	kernel_launch_timed_out              = CUDA_ERROR_LAUNCH_TIMEOUT,
	kernel_uses_incompatible_texturing_mode
	                                     = CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
	peer_access_already_enabled          = CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
	peer_access_already_disabled         = CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
	primary_context_already_active       = CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
	context_is_destroyed                 = CUDA_ERROR_CONTEXT_IS_DESTROYED,
	assert                               = CUDA_ERROR_ASSERT,
	too_many_peers                       = CUDA_ERROR_TOO_MANY_PEERS,
	host_memory_already_registered       = CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
	host_memory_not_registered           = CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
	hardware_stack_error                 = CUDA_ERROR_HARDWARE_STACK_ERROR,
	illegal_instruction                  = CUDA_ERROR_ILLEGAL_INSTRUCTION,
	misaligned_address                   = CUDA_ERROR_MISALIGNED_ADDRESS,
	invalid_address_space                = CUDA_ERROR_INVALID_ADDRESS_SPACE,
	invalid_pc                           = CUDA_ERROR_INVALID_PC,
	exception_during_kernel_execution    = CUDA_ERROR_LAUNCH_FAILED, // Not clear whether it's during scheduling, launch or execution
	cooperative_kernel_launch_grid_too_large
	                                     = CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
#if CUDA_VERSION >= 12800
	kernel_allocated_tensor_memory_not_freed
	                                     = CUDA_ERROR_TENSOR_MEMORY_LEAK,
#endif
	not_permitted                        = CUDA_ERROR_NOT_PERMITTED,
	not_supported                        = CUDA_ERROR_NOT_SUPPORTED,
#if CUDA_VERSION >= 10000
	system_not_ready                     = CUDA_ERROR_SYSTEM_NOT_READY,
#endif
#if CUDA_VERSION >= 10100
	system_cuda_and_display_drivers_mismatch
	                                     = CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
#endif
	not_supported_on_device              = CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE,
#if CUDA_VERSION >= 11040
	failure_connecting_to_mps_server     = CUDA_ERROR_MPS_CONNECTION_FAILED,
	mps_client_rpc_call_failed           = CUDA_ERROR_MPS_RPC_FAILURE,
	mps_server_not_ready_for_new_client_request
	                                     = CUDA_ERROR_MPS_SERVER_NOT_READY,
	resources_to_create_mps_clients_exhausted
	                                     = CUDA_ERROR_MPS_MAX_CLIENTS_REACHED,
	resources_for_mps_device_connections_exhausted
	                                     = CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED,
#endif
	mps_client_terminated_by_server      = CUDA_ERROR_MPS_CLIENT_TERMINATED,
	dynamic_parallelism_not_supported    = CUDA_ERROR_CDP_NOT_SUPPORTED,
	dynamic_parallelism_version_mismatch = CUDA_ERROR_CDP_VERSION_MISMATCH,
#if CUDA_VERSION >= 10000
	stream_capture_unsupported           = CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
	stream_capture_invalidated           = CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
	stream_capture_merge                 = CUDA_ERROR_STREAM_CAPTURE_MERGE,
	stream_capture_unmatched             = CUDA_ERROR_STREAM_CAPTURE_UNMATCHED,
	stream_capture_unjoined              = CUDA_ERROR_STREAM_CAPTURE_UNJOINED,
	stream_capture_isolation             = CUDA_ERROR_STREAM_CAPTURE_ISOLATION,
	stream_capture_disallowed_implicit_dependency
	                                     = CUDA_ERROR_STREAM_CAPTURE_IMPLICIT,
	not_permitted_on_captured_event      = CUDA_ERROR_CAPTURED_EVENT,
#endif
#if CUDA_VERSION >= 10100
	wrong_thread_for_ending_stream_capture
	                                     = CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD,
#endif
#if CUDA_VERSION >= 10200
	specified_timeout_lapsed             = CUDA_ERROR_TIMEOUT,
	graph_update_would_violate_constraints
	                                     = CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE,
#endif
#if CUDA_VERSION >= 11040
	async_error_on_external_device       = CUDA_ERROR_EXTERNAL_DEVICE,
#endif // CUDA_VERSION >= 11040
#if CUDA_VERSION >= 12000
	invalid_cluster_size                 = CUDA_ERROR_INVALID_CLUSTER_SIZE,
#endif // CUDA_VERSION >= 12000
#if CUDA_VERSION >= 12040
	necessary_function_not_loaded        = CUDA_ERROR_FUNCTION_NOT_LOADED,
	invalid_resource_type                = CUDA_ERROR_INVALID_RESOURCE_TYPE,
	resources_insufficient_or_inapplicable = CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION,
#endif // CUDA_VERSION >= 12040
#if CUDA_VERSION >= 12080
	key_rotation_sequence_failure        = CUDA_ERROR_KEY_ROTATION,
#endif // CUDA_VERSION >= 12080
#if CUDA_VERSION >= 13010
	not_permitted_on_detached_stream     = CUDA_ERROR_STREAM_DETACHED,
#endif
#if CUDA_VERSION >= 13030
	graph_recapture_failure              = CUDA_ERROR_GRAPH_RECAPTURE_FAILURE,
#endif
	unknown                              = CUDA_ERROR_UNKNOWN,

	// Other errors, only defined in the Runtime API headers.
	// TODO: Consider adding named constants for more of these

	invalid_configuration                = cudaErrorInvalidConfiguration,

	invalid_pitch_value                  = cudaErrorInvalidPitchValue,
	invalid_symbol                       = cudaErrorInvalidSymbol,
	invalid_host_pointer                 = cudaErrorInvalidHostPointer,
	invalid_device_pointer               = cudaErrorInvalidDevicePointer,
	invalid_texture                      = cudaErrorInvalidTexture,
	invalid_texture_binding              = cudaErrorInvalidTextureBinding,
	invalid_channel_descriptor           = cudaErrorInvalidChannelDescriptor,
	invalid_memcpy_direction             = cudaErrorInvalidMemcpyDirection,
	address_of_constant                  = cudaErrorAddressOfConstant,
	texture_fetch_failed                 = cudaErrorTextureFetchFailed,
	texture_not_bound                    = cudaErrorTextureNotBound,
	synchronization_error                = cudaErrorSynchronizationError,
	invalid_filter_setting               = cudaErrorInvalidFilterSetting,
	invalid_norm_setting                 = cudaErrorInvalidNormSetting,
	mixed_device_execution               = cudaErrorMixedDeviceExecution,
	not_yet_implemented                  = cudaErrorNotYetImplemented,
	memory_value_too_large               = cudaErrorMemoryValueTooLarge,

	driver_too_old_for_cuda_runtime      = cudaErrorInsufficientDriver,
	invalid_surface                      = cudaErrorInvalidSurface,
	prior_launch_failure                 = cudaErrorPriorLaunchFailure,
	invalid_kernel_function              = cudaErrorInvalidDeviceFunction,
	devices_unavailable                  = cudaErrorDevicesUnavailable,

	missing_configuration                = cudaErrorMissingConfiguration,

	launch_max_depth_exceeded            = cudaErrorLaunchMaxDepthExceeded,
	launch_file_scoped_tex               = cudaErrorLaunchFileScopedTex,
	launch_file_scoped_surfaces          = cudaErrorLaunchFileScopedSurf,
	sync_depth_exceeded                  = cudaErrorSyncDepthExceeded,
	launch_pending_count_exceeded        = cudaErrorLaunchPendingCountExceeded,

	invalid_device_function              = cudaErrorInvalidDeviceFunction,

	runtime_startup_failure              = cudaErrorStartupFailure,

#if CUDA_VERSION >= 10200
	device_not_initialized_for_runtime   = cudaErrorDeviceUninitialized, // see also not_yet_initialized
#endif

	set_on_active_process                = cudaErrorSetOnActiveProcess,
};

///@cond
constexpr bool operator==(const status_t& lhs, const named_t& rhs) noexcept { return lhs == static_cast<status_t>(rhs); }
constexpr bool operator!=(const status_t& lhs, const named_t& rhs) noexcept { return lhs != static_cast<status_t>(rhs); }
constexpr bool operator==(const named_t& lhs, const status_t& rhs) noexcept { return static_cast<status_t>(lhs) == rhs; }
constexpr bool operator!=(const named_t& lhs, const status_t& rhs) noexcept { return static_cast<status_t>(lhs) != rhs; }
///@endcond

} // namespace status

/// Determine whether the API call returning the specified status had succeeded
///@{
constexpr bool is_success(status_t status)  { return status == static_cast<status_t>(status::success); }
constexpr bool is_success(cudaError_t status) { return static_cast<status_t>(status) == static_cast<status_t>(status::success); }
///@}

/// @brief Determine whether the API call returning the specified status had failed
///@{
constexpr bool is_failure(status_t status)  { return not is_success(status); }
constexpr bool is_failure(cudaError_t status)  { return is_failure(static_cast<status_t>(status)); }
///@}

/// Obtain a brief textual explanation for a specified kind of CUDA Runtime API status or error code.
///@{
inline std::string describe(status_t status)
{
	// Even though status_t aliases the driver's CUresult type, some values are actually
	// runtime error codes. The driver will fail to identify them (they're luckily distinct),
	// and we can't distinguish proper failure from the case of a Runtime-API-only error
	// code - so we also try the runtime API.
	const char* description;
	auto description_lookup_status = cuGetErrorString(status, &description);
	return (description_lookup_status == CUDA_SUCCESS) ?
		description : cudaGetErrorString(static_cast<cudaError_t>(status));
}
inline std::string describe(cudaError_t status) { return cudaGetErrorString(status); }
///@}

namespace detail_ {

template <typename I, bool UpperCase = false>
std::string as_hex(I x)
{
	static_assert(std::is_unsigned<I>::value, "only signed representations are supported");
	unsigned num_hex_digits = 2*sizeof(I);
	if (x == 0) return "0x0";

	enum { bits_per_hex_digit = 4 }; // = log_2 of 16
	static const char* digit_characters =
		UpperCase ? "0123456789ABCDEF" : "0123456789abcdef" ;

	std::string result(num_hex_digits,'0');
	for (unsigned digit_index = 0; digit_index < num_hex_digits ; digit_index++)
	{
		size_t bit_offset = (num_hex_digits - 1 - digit_index) * bits_per_hex_digit;
		auto hexadecimal_digit = (x >> bit_offset) & 0xF;
		result[digit_index] = digit_characters[hexadecimal_digit];
	}
	return "0x0" + result.substr(result.find_first_not_of('0'), std::string::npos);
}

// TODO: Perhaps find a way to avoid the extra function, so that as_hex() can
// be called for pointer types as well? Would be easier with boost's uint<T>...
template <typename I, bool UpperCase = false>
std::string ptr_as_hex(const I* ptr)
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
class runtime_error : public std::runtime_error {
public:
	///@cond
	runtime_error(status_t error_code) :
		std::runtime_error(describe(error_code)), code_(error_code)
	{ }
	// I wonder if I should do this the other way around
	runtime_error(status_t error_code, const std::string& what_arg) :
		std::runtime_error(what_arg + ": " + describe(error_code)),
		code_(error_code)
	{ }
	// I wonder if I should do this the other way around
	runtime_error(status_t error_code, std::string&& what_arg) :
		runtime_error(error_code, what_arg)
	{ }
	///@endcond
	explicit runtime_error(status::named_t error_code) :
		runtime_error(static_cast<status_t>(error_code)) { }
	runtime_error(status::named_t error_code, const std::string& what_arg) :
		runtime_error(static_cast<status_t>(error_code), what_arg) { }
	runtime_error(status::named_t error_code, std::string&& what_arg) :
		runtime_error(static_cast<status_t>(error_code), what_arg) { }

protected:
	runtime_error(status_t error_code, std::runtime_error&& err) :
		std::runtime_error(std::move(err)), code_(error_code)
	{ }

public:
	/// Construct a runtime error which will not produce the default description for the error code,
	/// but rather only the specified message.
	static runtime_error with_message_override(status_t error_code, std::string complete_what_arg)
	{
		return runtime_error(error_code, std::runtime_error(std::move(complete_what_arg)));
	}

	/// Obtain the CUDA status code which resulted in this error being thrown.
	status_t code() const { return code_; }

private:
	status_t code_;
};

/// A macro for only throwing an error if we've failed - which also ensures no string
/// is constructed unless we actually need to throw
#define throw_if_error_lazy(status__, ... ) \
do { \
	const ::cuda_::status_t tie_status__ = static_cast<::cuda_::status_t>(status__); \
	if (::cuda_::is_failure(tie_status__)) { \
		throw ::cuda_::runtime_error(tie_status__, (__VA_ARGS__)); \
	} \
} while(false)

/**
 * Do nothing... unless the status indicates an error, in which case
 * a @ref cuda_::runtime_error exception is thrown
 *
 * @note Using these functions means the string will (almost certainly) be constructed,
 * hence you might want to use the @ref throw_if_error_lazy macro instead
 *
 * @param status should be @ref status::success  - otherwise an exception is thrown
 * @param message An extra description message to add to the exception
 */
///@{
inline void throw_if_error(status_t status, const std::string& message) noexcept(false)
{
	if (is_failure(status)) { throw runtime_error(status, message); }
}

inline void throw_if_error(cudaError_t status, const std::string& message) noexcept(false)
{
	throw_if_error(static_cast<status_t>(status), message);
}

inline void throw_if_error(status_t status, std::string&& message) noexcept(false)
{
	if (is_failure(status)) { throw runtime_error(status, message); }
}

inline void throw_if_error(cudaError_t status, std::string&& message) noexcept(false)
{
	return throw_if_error(static_cast<status_t>(status), message);
}
///@}

/**
 * Does nothing - unless the status indicates an error, in which case
 * a @ref cuda_::runtime_error exception is thrown
 *
 * @note Using these functions means the string will (almost certainly) be constructed,
 * hence you might want to use the @ref throw_if_error_lazy macro instead
*
 * @param status should be @ref cuda_::status::success - otherwise an exception is thrown
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
inline status_t get(bool try_clearing = false) noexcept(true)
{
	static constexpr unsigned dummy_flags{0};
	auto status = cuInit(dummy_flags);
	if (not is_success(status)) { return status; }
	return static_cast<status_t>(try_clearing ? cudaGetLastError() : cudaPeekAtLastError());
}

/**
 * @brief Does nothing (unless throwing an exception)
 *
 * @note Invoking this function will both make an API call and guarantee the construction
 * of the string message, regardless of whether an error has occurred, so it doesn't
 * quite do "nothing".
 *
 * @note similar to @ref cuda_::throw_if_error, but uses the CUDA driver's
 * own state regarding whether or not a sticky error has occurred
 */
inline void ensure_none(const std::string &message) noexcept(false)
{
	auto status = get();
	throw_if_error(status, message);
}

/**
 * @brief A variant of @ref ensure_none() which takes a C-style string.
 *
 * @note exists so as to avoid incorrect overload resolution of
 * `ensure_none(my_c_string)` calls.
 */
inline void ensure_none(const char *message) noexcept(false)
{
	return ensure_none(std::string{message});
}

/**
 * @brief Does nothing (except possibly throwing an exception)
 *
 * @note similar to @ref throw_if_error, but uses the CUDA Runtime API's internal
 * state
 *
 * @throws cuda_::runtime_error if the CUDA runtime API has encountered previously
 * encountered an (uncleared) error
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
inline std::string identify(device::id_t device_id)
{
	return std::string("device ") + std::to_string(device_id);
}
} // namespace detail_
} // namespace device

namespace context {
namespace detail_ {

inline std::string identify(handle_t handle)
{
	return "context " + cuda_::detail_::ptr_as_hex(handle);
}

inline std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail_::identify(device_id);
}

} // namespace detail_

namespace current {
namespace detail_ {
inline std::string identify(context::handle_t handle)
{
	return "current context: " + context::detail_::identify(handle);
}
inline std::string identify(context::handle_t handle, device::id_t device_id)
{
	return "current context: " + context::detail_::identify(handle, device_id);
}
} // namespace detail_
} // namespace current

} // namespace context

namespace device {
namespace primary_context {
namespace detail_ {

inline std::string identify(handle_t handle, device::id_t device_id)
{
	return "context " + context::detail_::identify(handle, device_id);
}
inline std::string identify(handle_t handle)
{
	return "context " + context::detail_::identify(handle);
}
} // namespace detail_
} // namespace primary_context
} // namespace device

namespace stream {
namespace detail_ {
inline std::string identify(handle_t handle)
{
	return (handle == nullptr) ? "default/null stream" :
		"stream at" + cuda_::detail_::ptr_as_hex(handle);
}
inline std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail_::identify(device_id);
}
inline std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " in " + context::detail_::identify(context_handle);
}
inline std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " in " + context::detail_::identify(context_handle, device_id);
}
} // namespace detail_
} // namespace stream

namespace event {
namespace detail_ {
inline std::string identify(handle_t handle)
{
	return "event " + cuda_::detail_::ptr_as_hex(handle);
}
inline std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail_::identify(device_id);
}
inline std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " on " + context::detail_::identify(context_handle);
}
inline std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " on " + context::detail_::identify(context_handle, device_id);
}
} // namespace detail_
} // namespace event

namespace kernel {
namespace detail_ {

inline std::string identify(const void* ptr)
{
	return "kernel " + cuda_::detail_::ptr_as_hex(ptr);
}
inline std::string identify(const void* ptr, device::id_t device_id)
{
	return identify(ptr) + " on " + device::detail_::identify(device_id);
}
inline std::string identify(const void* ptr, context::handle_t context_handle)
{
	return identify(ptr) + " in " + context::detail_::identify(context_handle);
}
inline std::string identify(const void* ptr, context::handle_t context_handle, device::id_t device_id)
{
	return identify(ptr) + " in " + context::detail_::identify(context_handle, device_id);
}
inline std::string identify(handle_t handle)
{
	return "kernel at " + cuda_::detail_::ptr_as_hex(handle);
}
inline std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " in " + context::detail_::identify(context_handle);
}
inline std::string identify(handle_t handle,  device::id_t device_id)
{
	return identify(handle) + " on " + device::detail_::identify(device_id);
}
inline std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " in " + context::detail_::identify(context_handle, device_id);
}

} // namespace detail_
} // namespace kernel

namespace memory {
namespace detail_ {

inline std::string identify(region_t region)
{
	return std::string("memory region at ") + cuda_::detail_::ptr_as_hex(region.data())
		+ " of size " + std::to_string(region.size());
}

inline std::string identify(location_t location)
{
	switch (location.type) {
	case CU_MEM_LOCATION_TYPE_DEVICE:
		if (location.id != CU_DEVICE_CPU) {
			return "global memory of " + cuda_::device::detail_::identify(location.id);
		}
		// fallthrough
	case CU_MEM_LOCATION_TYPE_HOST:
		return "host (system) memory";
	case CU_MEM_LOCATION_TYPE_HOST_NUMA:
		return "host (system) NUMA node " + std::to_string(location.id);
	case CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT:
		return "current host (system) NUMA node ";
	default:
		return "(invalid)";
	}
}

} // namespace detail_

} // namespace memory

} // namespace cuda_

#endif // CUDA_API_WRAPPERS_ERROR_HPP_
