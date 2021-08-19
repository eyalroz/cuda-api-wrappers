/**
 * @file api/device_properties.hpp
 *
 * @brief Classes for holding CUDA device properties and
 * CUDA compute capability values.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_DEVICE_PROPERTIES_HPP_
#define CUDA_API_WRAPPERS_DEVICE_PROPERTIES_HPP_

#include <cuda/api/constants.hpp>
#include <cuda/api/pci_id.hpp>
#include <cuda/common/types.hpp>

#include <cuda_runtime_api.h>

#include <stdexcept>


// The following un-definitions avoid warnings about
// the use of `major` and `minor` in certain versions
// of the GNU C library
#ifdef major
#undef major
#endif

#ifdef minor
#undef minor
#endif

namespace cuda {

namespace device {

/**
 * A numeric designator of an architectural generation of CUDA devices
 *
 * @note See <a href="https://en.wikipedia.org/wiki/Category:Nvidia_microarchitectures">this listing</a>
 * of nVIDIA GPU microarchitectures. Also see @ref compute_capability_t .
 */
struct compute_architecture_t {
	/**
	 * A @ref compute_capability_t has a "major" and a "minor" number,
	 * with "major" indicating the architecture; so this struct only
	 * has a "major" number
	 */
	unsigned major;

	const char* name() const;
	unsigned max_warp_schedulings_per_processor_cycle() const;
	unsigned max_resident_warps_per_processor() const;
	unsigned max_in_flight_threads_per_processor() const;

	/**
	 * @note On some architectures, the shared memory / L1 balance is configurable,
	 * so you might not get the maxima here without making this configuration
	 * setting
	 */
	memory::shared::size_t max_shared_memory_per_block() const;

	constexpr bool is_valid() const noexcept;

};

// TODO: Consider making this a non-POD struct,
// with a proper ctor checking validity, an operator converting to pair etc;
// however, that would require including at least ::std::utility, if not other
// stuff (e.g. for an ::std::hash specialization)
// TODO: If we constrained this to versions we know about, we could make the
// methods noexcept
/**
 * A numeric designator of the computational capabilities of a CUDA device
 *
 * @note Wikipedia has a <a href="https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications"</a>table</a>
 * listing the specific features and capabilities for different CC values.
 */
struct compute_capability_t {

	compute_architecture_t architecture;
	unsigned minor_;

	constexpr static compute_capability_t from_combined_number(unsigned combined) noexcept;

	constexpr unsigned major() const { return architecture.major; }
	unsigned constexpr minor() const { return minor_; }

	constexpr unsigned as_combined_number() const noexcept;
	constexpr bool is_valid() const noexcept;
	unsigned max_warp_schedulings_per_processor_cycle() const;
	unsigned max_resident_warps_per_processor() const;
	unsigned max_in_flight_threads_per_processor() const;
	/**
	 * @note On some architectures, the shared memory / L1 balance is configurable,
	 * so you might not get the maxima here without making this configuration
	 * setting
	 */
	memory::shared::size_t max_shared_memory_per_block() const;


};

/**
 * @brief A named constructor idiom for {@ref compute_capability_t}
 *
 * @param combined A combination of the major and minor version number, e.g. 91 for major 9, minor 1
 */
constexpr compute_capability_t make_compute_capability(unsigned combined) noexcept;

/**
 * @brief A named constructor idiom for {@ref compute_capability_t}.
 */
constexpr compute_capability_t make_compute_capability(unsigned major, unsigned minor) noexcept;

/**
 * @brief A structure holding a collection various properties of a device
 *
 * @note Somewhat annoyingly, CUDA devices have attributes, properties and flags.
 * Attributes have integral number values; properties have all sorts of values,
 * including arrays and limited-length strings (see
 * @ref cuda::device::properties_t), and flags are either binary or
 * small-finite-domain type fitting into an overall flagss value (see
 * @ref cuda::device_t::flags_t). Flags and properties are obtained all at once,
 * attributes are more one-at-a-time.
 *
 */
struct properties_t : public cudaDeviceProp {

	properties_t() = default;
	properties_t(const cudaDeviceProp& cdp) noexcept : cudaDeviceProp(cdp) { };
	properties_t(cudaDeviceProp&& cdp) noexcept : cudaDeviceProp(cdp) { };
	bool usable_for_compute() const noexcept;
	compute_capability_t compute_capability() const noexcept { return { { (unsigned) major }, (unsigned) minor }; }
	compute_architecture_t compute_architecture() const noexcept { return { (unsigned) major }; };
	pci_location_t pci_id() const noexcept { return { pciDomainID, pciBusID, pciDeviceID }; }

	unsigned long long max_in_flight_threads_on_device() const
	{
		return compute_capability().max_in_flight_threads_per_processor() * multiProcessorCount;
	}

	grid::block_dimension_t max_threads_per_block() const noexcept { return maxThreadsPerBlock; }
	grid::block_dimension_t max_warps_per_block() const noexcept { return maxThreadsPerBlock / warp_size; }
	size_t max_shared_memory_per_block() const noexcept { return sharedMemPerBlock; }
	size_t global_memory_size() const noexcept { return totalGlobalMem; }
	bool can_map_host_memory() const noexcept { return canMapHostMemory != 0; }
};

} // namespace device
} // namespace cuda

#include <cuda/api/detail/device_properties.hpp>

#endif // CUDA_API_WRAPPERS_DEVICE_PROPERTIES_HPP_
