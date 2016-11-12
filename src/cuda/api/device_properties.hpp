#pragma once
#ifndef CUDA_DEVICE_PROPERTIES_HPP_
#define CUDA_DEVICE_PROPERTIES_HPP_

#include "cuda/api/types.h"
#include "cuda/api/constants.h"

#include <cuda_runtime_api.h>

namespace cuda {

namespace device {

const char* architecture_name(unsigned major_compute_capability_version);

// TODO: Consider making this a non-POD struct,
// with a proper ctor checking validity, an operator converting to pair etc;
// however, that would require including at least std::utility, if not other
// stuff (e.g. for an std::hash specialization)
struct compute_capability_t {
	unsigned major;
	unsigned minor;

	unsigned as_combined_number() const { return major * 10 + minor; }
	unsigned max_warp_schedulings_per_processor_cycle() const;
	unsigned max_resident_warps_per_processor() const;
	unsigned max_in_flight_threads_per_processor() const;
	shared_memory_size_t max_shared_memory_per_block() const;
	const char* architecture_name() const
	{
		return device::architecture_name(major);
	}

	static compute_capability_t from_combined_number(unsigned combined)
	{
		return  { combined / 10, combined % 10 };
	}
	bool is_valid() const
	{
		return (major > 0) and (major < 9999) and (minor > 0) and (minor < 9999);
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

struct properties_t : public cudaDeviceProp {

	properties_t() = default;
	properties_t(cudaDeviceProp& cdp) : cudaDeviceProp(cdp) { };

	bool usable_for_compute() const
	{
		return computeMode != cudaComputeModeProhibited;
	}

	compute_capability_t compute_capability() const {
		return {(unsigned) major, (unsigned) minor};
	}
	operator compute_capability_t() const {
		return compute_capability();
	}

	unsigned long long max_in_flight_threads_on_device()
	{
		return compute_capability().max_in_flight_threads_per_processor() * multiProcessorCount;
	}

	/**
	 * The CUDA device properties have a field with the supposed maximum amount of shared
	 * memory per block you can use, but - that field is a dirty lie, is what it is!
	 * On Kepler, you can only get 48 KiB shared memory and the rest is always reserved
	 * for L1...
	 * so, if you call this function, you'll get the proper effective maximum figure.
	 */
	const char*  architecture_name() { return compute_capability().architecture_name(); }

	/**
	 * A convenience method which applies an appropriate cast (e.g. for avoiding
	 * signed/unsigned comparison warnings)
	 */
	grid_block_dimension_t max_threads_per_block() { return maxThreadsPerBlock; }

	grid_block_dimension_t max_warps_per_block() { return maxThreadsPerBlock / warp_size; }

};

} // namespace device
} // namespace cuda

#endif /* CUDA_DEVICE_PROPERTIES_HPP_ */
