/**
 * @file device_properties.hpp
 *
 * @brief Classes for holding CUDA device properties and
 * CUDA compute capability values.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_DEVICE_PROPERTIES_HPP_
#define CUDA_API_WRAPPERS_DEVICE_PROPERTIES_HPP_

#include <cuda/api/types.h>
#include <cuda/api/constants.h>
#include <cuda/api/pci_id.h>

#include <cuda_runtime_api.h>

namespace cuda {

namespace device {

/**
 * A numeric designator of an architectural generation of CUDA devices
 *
 * @note See @url https://en.wikipedia.org/wiki/Volta_(microarchitecture)
 * and previous architectures' pages via "previous" links.
 * Also see @ref compute_capability_t .
 */
struct compute_architecture_t {
	/**
	 * A @ref compute_capability_t has a "major" and a "minor" number,
	 * with "major" indicating the architecture; so this struct only
	 * has a "major" numner
	 */
	unsigned major;

	static const char* name(unsigned major_compute_capability_version);
	unsigned max_warp_schedulings_per_processor_cycle() const;
	unsigned max_resident_warps_per_processor() const;
	unsigned max_in_flight_threads_per_processor() const;
	/**
	 * @note On some architectures, the shared memory / L1 balance is configurable,
	 * so you might not get the maxima here without making this configuration
	 * setting
	 */
	shared_memory_size_t max_shared_memory_per_block() const;
	const char* name() const { return name(major); }

	bool is_valid() const
	{
		return (major > 0) and (major < 9999); // Picked this up from the CUDA code somwhere
	}

};

inline bool operator ==(const compute_architecture_t& lhs, const compute_architecture_t& rhs)
{
	return lhs.major == rhs.major;
}
inline bool operator !=(const compute_architecture_t& lhs, const compute_architecture_t& rhs)
{
	return lhs.major != rhs.major;
}
inline bool operator <(const compute_architecture_t& lhs, const compute_architecture_t& rhs)
{
	return lhs.major < rhs.major;
}
inline bool operator <=(const compute_architecture_t& lhs, const compute_architecture_t& rhs)
{
	return lhs.major < rhs.major;
}
inline bool operator >(const compute_architecture_t& lhs, const compute_architecture_t& rhs)
{
	return lhs.major > rhs.major;
}
inline bool operator >=(const compute_architecture_t& lhs, const compute_architecture_t& rhs)
{
	return lhs.major > rhs.major;
}


// TODO: Consider making this a non-POD struct,
// with a proper ctor checking validity, an operator converting to pair etc;
// however, that would require including at least std::utility, if not other
// stuff (e.g. for an std::hash specialization)
/**
 * A numeric designator of the computational capabilities of a CUDA device
 *
 * @note See @url https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications
 * for a specification of capabilities by CC values
 */
struct compute_capability_t {
	unsigned major;
	unsigned minor;

	unsigned as_combined_number() const { return major * 10 + minor; }
	unsigned max_warp_schedulings_per_processor_cycle() const;
	unsigned max_resident_warps_per_processor() const;
	unsigned max_in_flight_threads_per_processor() const;
	/**
	 * @note On some architectures, the shared memory / L1 balance is configurable,
	 * so you might not get the maxima here without making this configuration
	 * setting
	 */
	shared_memory_size_t max_shared_memory_per_block() const;
	compute_architecture_t architecture() const { return compute_architecture_t { major }; }

	bool is_valid() const
	{
		return (major > 0) and (major < 9999) and (minor > 0) and (minor < 9999);
			// Picked this up from the CUDA code somwhere
	}

	static compute_capability_t from_combined_number(unsigned combined)
	{
		return  { combined / 10, combined % 10 };
	}
};

inline bool operator ==(const compute_capability_t& lhs, const compute_capability_t& rhs)
{
	return lhs.major == rhs.major and lhs.minor == rhs.minor;
}
inline bool operator !=(const compute_capability_t& lhs, const compute_capability_t& rhs)
{
	return lhs.major != rhs.major or lhs.minor != rhs.minor;
}
inline bool operator <(const compute_capability_t& lhs, const compute_capability_t& rhs)
{
	return lhs.major < rhs.major or (lhs.major == rhs.major and lhs.minor < rhs.minor);
}
inline bool operator <=(const compute_capability_t& lhs, const compute_capability_t& rhs)
{
	return lhs.major < rhs.major or (lhs.major == rhs.major and lhs.minor <= rhs.minor);
}
inline bool operator >(const compute_capability_t& lhs, const compute_capability_t& rhs)
{
	return lhs.major > rhs.major or (lhs.major == rhs.major and lhs.minor > rhs.minor);
}
inline bool operator >=(const compute_capability_t& lhs, const compute_capability_t& rhs)
{
	return lhs.major > rhs.major or (lhs.major == rhs.major and lhs.minor >= rhs.minor);
}

inline compute_capability_t make_compute_capability(unsigned combined)
{
	return compute_capability_t::from_combined_number(combined);
}

inline compute_capability_t make_compute_capability(unsigned major, unsigned minor)
{
	return { major, minor };
}

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
	properties_t(cudaDeviceProp& cdp) : cudaDeviceProp(cdp) { };
	bool usable_for_compute() const
	{
		return computeMode != cudaComputeModeProhibited;
	}
	compute_capability_t compute_capability() const { return { (unsigned) major, (unsigned) minor }; }
	compute_capability_t compute_architecture() const { return { (unsigned) major }; };
	pci_location_t pci_id() const { return { pciDomainID, pciBusID, pciDeviceID }; }

	unsigned long long max_in_flight_threads_on_device() const
	{
		return compute_capability().max_in_flight_threads_per_processor() * multiProcessorCount;
	}

	grid_block_dimension_t max_threads_per_block() const { return maxThreadsPerBlock; }
	grid_block_dimension_t max_warps_per_block() const { return maxThreadsPerBlock / warp_size; }
	bool can_map_host_memory() const { return canMapHostMemory != 0; }
};

} // namespace device
} // namespace cuda

#endif /* CUDA_API_WRAPPERS_DEVICE_PROPERTIES_HPP_ */
