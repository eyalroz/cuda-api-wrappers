/**
 * @file
 *
 * @brief Classes representing specific and overall properties of CUDA devices.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_DEVICE_PROPERTIES_HPP_
#define CUDA_API_WRAPPERS_DEVICE_PROPERTIES_HPP_

#include "constants.hpp"
#include "pci_id.hpp"

#include "types.hpp"

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

/// Type of the number of mutiprocessors within a single GPU.
using multiprocessor_count_t = int;

/**
 * A numeric designator of an architectural generation of CUDA devices
 *
 * @note See <a href="https://en.wikipedia.org/wiki/Category:Nvidia_microarchitectures">this listing</a>
 * of nVIDIA GPU microarchitectures; cf. @ref compute_capability_t .
 */
struct compute_architecture_t {
	/**
	 * A @ref compute_capability_t has a "major" and a "minor" number,
	 * with "major" indicating the architecture; so this struct only
	 * has a "major" number
	 */
	unsigned major;

	/**
	 * @returns the name NVIDIA has given this microarchitecture
	 *
	 * @note NVIDIA names their microarchitecture after famous scientists  like "Tesla", "Pascal" etc.
	 */
	const char* name() const;

	/// @return true if @ref major is indeed a number of a known/recognized NVIDIA GPU
	/// microarchitecture.
	constexpr bool is_valid() const noexcept;

};

/**
 * A numeric designator of the computational capabilities of a CUDA device
 *
 * @note The CUDA programming guide has tables
 * (<a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-feature-support-per-compute-capability"</a>this one</a>
 * and
 * <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability"</a>this one</a>)
 * listing the specific features and capabilities for different CC values.
 */
struct compute_capability_t {

	/// The major capability designator
	compute_architecture_t architecture;

	/// The minor designator, indicating mostly numeric choices of capabilities (e.g. how many SMs,
	/// homw much memory, whether the numbers of functional units will be skewed more towards double-precision or
	/// integer operations etc.)
	unsigned minor_;

	/**
	 * Converts a single-number representation of a compute capability into a proper structured instance
	 * of this class.
	 *
	 * In certain contexts (e.g. compiler command-line parameters), compute capabilities are specified
	 * by a single number, e.g. 75 for major 7, minor 5. This perform one direction of the conversion; see
	 * also {@ref as_combined_number}.
	 */
	constexpr static compute_capability_t from_combined_number(unsigned combined) noexcept;

	constexpr unsigned major() const { return architecture.major; }
	unsigned constexpr minor() const { return minor_; }

	/**
	 * Produces a single-number representation of the compute capability.
	 *
	 * In certain contexts (e.g. compiler command-line parameters), compute capabilities are specified
	 * by a single number, e.g. 75 for major 7, minor 5. This perform one direction of the conversion; see
	 * also {@ref from_combined_number}.
	 */
	constexpr unsigned as_combined_number() const noexcept;

	/// @return true if there actually are any GPUs listed with this combination of major
	/// and minor compute capability numbers
	constexpr bool is_valid() const noexcept;

	/// Some (of many) specific properties of GPUs with a given compute capability
	///@{
	unsigned max_warp_schedulings_per_processor_cycle() const;
	unsigned max_resident_warps_per_processor() const;
	///@}

	///@note: Based on _ConvertSMVer2Cores() in the CUDA samples helper code
	unsigned max_in_flight_threads_per_processor() const;

	/// @note On some architectures, the shared memory / L1 balance is configurable, so that this
	/// may not be the current, actual maximum a specific kernel can use at a specific point.
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
 * @note Somewhat annoyingly, CUDA devices have all of attributes, properties and flags.
 * Attributes have integral number values; properties have all sorts of values,
 * including arrays and limited-length strings, and flags are actually associated with
 * a device's primary context, as it is actually _contexts_ which have flags (which are
 * either binary or small-finite-domain type fitting into an overall flags value:
 * {@ref context::flags_t}). Flags and properties are obtained all at once (the latter,
 * using the runtime API),
 * attributes are more one-at-a-time.
 */
struct properties_t : public cudaDeviceProp {

	properties_t() = default;
	properties_t(const cudaDeviceProp& cdp) noexcept : cudaDeviceProp(cdp) { };
	properties_t(cudaDeviceProp&& cdp) noexcept : cudaDeviceProp(cdp) { };

	/// Convenience methods for accessing fields in the raw parent class, or combining or converting
	/// them into a more useful type
	///@{
	bool usable_for_compute() const noexcept;
	compute_capability_t compute_capability() const noexcept
	{
		return { { static_cast<unsigned>(major) }, static_cast<unsigned>(minor) };
	}
	compute_architecture_t compute_architecture() const noexcept { return { static_cast<unsigned>(major) }; };
	pci_location_t pci_id() const noexcept { return { pciDomainID, pciBusID, pciDeviceID, pci_location_t::unused }; }

	unsigned long long max_in_flight_threads_on_device() const
	{
		return compute_capability().max_in_flight_threads_per_processor() * multiProcessorCount;
	}

	grid::block_dimension_t max_threads_per_block() const noexcept { return maxThreadsPerBlock; }
	grid::block_dimension_t max_warps_per_block() const noexcept { return maxThreadsPerBlock / warp_size; }
	size_t max_shared_memory_per_block() const noexcept { return sharedMemPerBlock; }
	size_t global_memory_size() const noexcept { return totalGlobalMem; }
	bool can_map_host_memory() const noexcept { return canMapHostMemory != 0; }
	///@}
};

} // namespace device
} // namespace cuda

#include "detail/device_properties.hpp"

#endif // CUDA_API_WRAPPERS_DEVICE_PROPERTIES_HPP_
