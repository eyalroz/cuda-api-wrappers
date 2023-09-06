/**
 * @file
 *
 * @brief Complementing file for `cuda/api/versions.hpp` for obtaining
 * a version number for the NVML library
 *
 * @note This file does not include wrappers for
 * `nvmlSystemGetCudaDriverVersion()`,
 * since the CUDA driver API itself can be used for this functionality; see
 * @ref cuda::version_numbers::driver() .
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NVML_VERSIONS_HPP_
#define CUDA_API_WRAPPERS_NVML_VERSIONS_HPP_

#include "error.hpp"
#include "../api/versions.hpp"
#include <cstring>

namespace cuda {

struct driver_version_t {
    int x;
    int y;
    int z;
};

namespace version_numbers {

namespace detail_ {

driver_version_t destructive_parse_driver_version(char* driver_version_str) noexcept(false)
{
    driver_version_t result;
    auto first_dot = ::std::strchr(driver_version_str, '.');
    if (first_dot == nullptr) {
        throw ::std::logic_error("Unrecognized NVIDIA driver version format: "
            + ::std::string(driver_version_str));
    }
    *first_dot = '\0';
    result.x = ::std::atoi(driver_version_str);
    auto second_dot = ::std::strchr(first_dot + 1, '.');
    if (second_dot == nullptr) {
        throw ::std::logic_error("Unrecognized NVIDIA driver version format: "
            + ::std::string(driver_version_str));
    }
    *second_dot = '\0';
    result.y = ::std::atoi(first_dot + 1);
    result.z = ::std::atoi(second_dot + 1);
    return result;
}

} // namespace detail_

/**
 * Obtain the NVIDIA X.Y.Z-format driver version
 */
inline driver_version_t driver() {
    char version_buffer[NVML_DEVICE_PART_NUMBER_BUFFER_SIZE + 1];
    version_buffer[NVML_DEVICE_PART_NUMBER_BUFFER_SIZE] = '\0';
    auto status = nvmlSystemGetDriverVersion(version_buffer, sizeof(version_buffer) - 1);
    throw_if_error_lazy(status, "Failed obtaining the NVIDIA (graphics) driver  version");
    return detail_::destructive_parse_driver_version(version_buffer);
}

/**
 * Obtain the NVML library version
 *
 * @todo If the version is a dot-separated sequence of numbers, use an
 * appropriate structure, like `cuda::version_t`, instead of a allocating
 * a string on the stack
 */
inline ::std::string nvml() {
	char version_buffer[NVML_DEVICE_PART_NUMBER_BUFFER_SIZE + 1] = {};
	auto status = nvmlSystemGetNVMLVersion(version_buffer, sizeof(version_buffer) - 1);
	throw_if_error_lazy(status, "Failed obtaining the NVML library version");
	return version_buffer;
}

inline ::std::string process_name(unsigned int pid) {
    static constexpr const size_t max_process_name_length = 256;
    char process_name_buffer[max_process_name_length + 1];
    process_name_buffer[max_process_name_length] = '\0';
    auto status = nvmlSystemGetProcessName(pid, process_name_buffer, max_process_name_length);
    throw_if_error_lazy(status, "Failed obtaining the NVML library version");
    return process_name_buffer;
}

} // namespace version_numbers
} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVML_VERSIONS_HPP_
