/**
 * Derived from the nVIDIA CUDA 8.0 samples by
 *
 *   Eyal Rozenberg
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 */

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking

#ifndef HELPER_CUDA_HPP_
#define HELPER_CUDA_HPP_

#include "../common.hpp"

#include "helper_string.h"

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

void ensure_device_is_usable(const cuda::device_t device)
{
	auto properties = device.properties();

	if (not properties.usable_for_compute()) {
		die_("Error: device " + std::to_string(device.id()) + "is running with <Compute Mode Prohibited>.");
	}

	if (properties.compute_capability().major() < 1) {
		die_("CUDA device " + std::to_string(device.id()) + " does not support CUDA.\n");
	}
}

// This function returns the best GPU (with maximum GFLOPS)
inline int get_device_with_highest_gflops()
{
	auto device_count = cuda::device::count();

	if (device_count == 0) {
		die_("get_device_with_highest_gflops() CUDA error: no devices supporting CUDA.");
	}

	std::vector<cuda::device::id_t> device_ids(device_count);
	std::iota(device_ids.begin(), device_ids.end(), 0);
	(void) std::remove_if(device_ids.begin(), device_ids.end(),
		[](cuda::device::id_t id) {
			auto properties = cuda::device::get(id).properties();
			return !properties.usable_for_compute()
				or !properties.compute_capability().is_valid();
		}
	);
	if (device_ids.empty()) {
		die_("get_device_with_highest_gflops() CUDA error: all devices have compute mode prohibited.");
	}
	if (device_ids.size() == 1) { return *device_ids.begin(); }

	auto iterator = std::max_element(device_ids.begin(),device_ids.end(),
		[](cuda::device::id_t id_1, cuda::device::id_t id_2) {
			auto cc_1 = cuda::device::get(id_1).properties().compute_capability();
			auto cc_2 = cuda::device::get(id_2).properties().compute_capability();
			return cc_1 > cc_2;
		}
	);
	auto best_sm_arch =	cuda::device::get(*iterator).properties().major;
	if (best_sm_arch > 2) {
		(void) std::remove_if(device_ids.begin(), device_ids.end(),
			[best_sm_arch](cuda::device::id_t id) {
				return cuda::device::get(id).properties().compute_capability().major() < (unsigned) best_sm_arch;
			}
		);
	}

	auto performance_estimator = [](cuda::device::id_t device_id) {
		auto properties = cuda::device::get(device_id).properties();
		return properties.max_in_flight_threads_on_device() * properties.clockRate;
	};
	iterator = std::max_element(device_ids.begin(),device_ids.end(),
		[&performance_estimator](cuda::device::id_t id_1, cuda::device::id_t id_2) {
			return performance_estimator(id_1) < performance_estimator(id_2);
		}
	);
	return *iterator;
}


// Initialization code to find the best CUDA Device
// Unlike in NVIDIA's original helper_cuda.h, this does _not_
// make the chosen device current.
inline cuda::device_t chooseCudaDevice(int argc, const char **argv)
{
	// If the command-line has a device number specified, use it
	if (checkCmdLineFlag(argc, argv, "device"))
	{
		auto device_id = getCmdLineArgumentInt(argc, argv, "device=");
		if (device_id < 0) {
			die_("Invalid command line parameter");
		}
		auto device = cuda::device::get(device_id);
		ensure_device_is_usable(device);
		return device;
	}
	else
	{
		// Otherwise pick the device with highest Gflops/s
		auto best_device = cuda::device::get(get_device_with_highest_gflops());
		std::cout << "GPU Device " << best_device.id() << ": ";
		std::cout << "\"" << best_device.name() << "\" ";
		std::cout << "with compute capability " << best_device.properties().compute_capability() << "\n";
		return best_device;
	}
}



#endif // HELPER_CUDA_HPP_
