/**
 * @file
 *
 * @brief NVML library wrappers for interacting with individual devices,
 * sans device query APIs covered by {@ref device_queries.hpp}.
 *
 * @note This file does not include wrappers for
 * `nvmlSystemGetCudaDriverVersion()`,
 * since the CUDA driver API itself can be used for this functionality; see
 * @ref cuda::version_numbers::driver() .
 */

#ifndef CUDA_API_WRAPPERS_NVML_DEVICE_HPP_
#define CUDA_API_WRAPPERS_NVML_DEVICE_HPP_

#include "types.hpp"
#include "error.hpp"

#include <string>

namespace cuda {

namespace nvml {

namespace device {


// TODO: Should we, instead, use size_t? something else?
cuda::device::id_t count() {
	unsigned int raw_count;
	auto status = nvmlDeviceGetCount(&raw_count);
	throw_if_nvml_error_lazy(status, "Obtaining the system's GPU device count");
	return raw_count;
}

namespace detail_ {

inline ::std::string identify(handle_t handle)
{
	return "NVML device handle at " + cuda::detail_::ptr_as_hex(handle);
}

} // namespace detail_

namespace clock {

namespace detail_ {

inline CPP14_CONSTEXPR const char* name(clocked_entity_t kind)
{
	// Keep this in sync with the kind_t enum;
	static constexpr const char* names[] = { "graphics", "symmetric multiprocessors", "memory", "video" };
	return names[static_cast<int>(kind)];
}

inline ::std::string identify(clocked_entity_t kind)
{
	return ::std::string("device ") + name(kind) + " clock";
}

inline nvmlClockType_t raw_clock_type_of(clock::clocked_entity_t kind)
{
	return static_cast<nvmlClockType_t>(kind);
}

inline void reset(handle_t device_handle, clock::clocked_entity_t kind)
{
	using reset_function_type = decltype(&nvmlDeviceResetGpuLockedClocks);
	reset_function_type function;
	nvml::status_t status;
	switch(kind) {
	case clock::clocked_entity_t::sm :
		function = &nvmlDeviceResetGpuLockedClocks; break;
	case clock::clocked_entity_t::memory:
		function = &nvmlDeviceResetMemoryLockedClocks; break;
	default:
		throw ::std::invalid_argument("Resetting clocks of this kind is not supported... somehow");
	}
	status = function(device_handle);
	throw_if_nvml_error_lazy(status, "Resetting " + clock::detail_::identify(kind)
		 + " for " + device::detail_::identify(device_handle));
}

inline void reset(handle_t device_handle)
{
	reset(device_handle, clock::clocked_entity_t::memory);
	reset(device_handle, clock::clocked_entity_t::symmetric_multiprocessors);
}

inline void set_global(
	handle_t device_handle,
	clock::clocked_entity_t clocked_entity,
	clock::mhz_frequency_range_t range)
{
	using setter_function_type = decltype(&nvmlDeviceSetGpuLockedClocks);
	setter_function_type function;
	nvml::status_t status;
	switch (clocked_entity) {
	case clock::clocked_entity_t::sm :
		function = &nvmlDeviceSetGpuLockedClocks; break;
	case clock::clocked_entity_t::memory:
		function = &nvmlDeviceSetMemoryLockedClocks; break;
	default:
		throw ::std::invalid_argument("Globally setting clocks of this kind is not supported... somehow");
	}
	status = function(device_handle, range.min, range.max);
	throw_if_nvml_error_lazy(status, "Setting a range of values for " + clock::detail_::identify(clocked_entity)
									 + " of " + device::detail_::identify(device_handle));
}

inline void set_for_future_contexts(
	handle_t device_handle,
	mhz_frequency_t memory_frequency,
	mhz_frequency_t cores_frequency)
{
	auto status = nvmlDeviceSetApplicationsClocks(device_handle, memory_frequency, cores_frequency);
	throw_if_nvml_error_lazy(status,
		"Setting memory and cores (SM) frequencies for new contexts in this application to ("
		+ ::std::to_string(memory_frequency) + " MHz, " + ::std::to_string(cores_frequency)
		+ " MHz) for " + device::detail_::identify(device_handle));
}

inline clock::mhz_frequency_t get(
	device::handle_t device_handle,
	scope_t scope,
	clock::clocked_entity_t entity)
{
	// Not currently supporting "application clocks"
	clock::mhz_frequency_t result;
	auto status = nvmlDeviceGetClock(device_handle, raw_clock_type_of(entity), raw_clock_id_of(scope), &result);
	throw_if_nvml_error_lazy(status, "Obtaining the current value of " + clock::detail_::identify(entity)
		+ " for " + device::detail_::identify(device_handle));
}

} // namespace detail_
} // namespace clock

namespace detail_ {

// Note: The signature, and implementation, of this function assume that
// CUDA device IDs and NVML device indices are always exactly the same
inline handle_t get_handle(cuda::device::id_t id)
{
	handle_t result;
	auto status = nvmlDeviceGetHandleByIndex_v2(id, &result);
	throw_if_nvml_error_lazy(status,"Obtaining a handle for " + cuda::device::detail_::identify(id));
	return result;
}

} // namespace detail_

} // namespace device

} // namespace nvml

} // namespace cuda
#endif // CUDA_API_WRAPPERS_NVML_DEVICE_HPP_