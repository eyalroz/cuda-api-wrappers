#pragma once
#ifndef CUDA_DEVICE_PROPERTIES_HPP_
#define CUDA_DEVICE_PROPERTIES_HPP_

#include "cuda/api/types.h"
#include "cuda/api/constants.h"

#include <cuda_runtime_api.h>

namespace cuda {
namespace device {

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
