/**
 * Derived from the nVIDIA CUDA 8.0 samples by
 *
 *   Eyal Rozenberg <E.Rozenberg@cwi.nl>
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 */

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking

#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cuda/api_wrappers.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>

#include "helper_string.h"
//#include "error_messages.h"

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif


#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

[[noreturn]] inline void die_(const std::string& message, int exit_value = EXIT_FAILURE) noexcept
{
	std::cerr << message << "\n";
	exit(exit_value);
}

inline std::ostream& operator<< (std::ostream& os, const cuda::device::compute_capability_t& cc)
{
	return os << cc.major << '.' << cc.minor;
}

#ifdef __CUDA_RUNTIME_H__
// General GPU Device CUDA Initialization
inline int gpuDeviceInit(int device_id)
{
	auto device_count = cuda::device::count();

	if (device_count == 0) {
		die_ ("gpuDeviceInit() CUDA error: no devices supporting CUDA.");
	}

	device_id  = std::max(device_id, 0);

	if (device_id > device_count-1) {
		std::cerr
			<< "\n"
			<< ">> " << device_count << " CUDA capable GPU device(s) detected. <<\n"
			<< ">> gpuDeviceInit (-device=" << device_id << " is not a valid GPU device. <<\n";
		return -device_id;
	}

	auto device = cuda::device::get(device_id);
	auto properties = device.properties();

	if (!properties.usable_for_compute()) {
		die_("Error: device is running in <Compute Mode Prohibited>, "
			"no threads can use ::cudaSetDevice().\n");
	}

	if (properties.compute_capability().major < 1) {
		die_("gpuDeviceInit(): GPU device does not support CUDA.\n");
	}

	device.make_current();
	return device_id;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId()
{
	auto device_count = cuda::device::count();

	if (device_count == 0) {
		die_("gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA.");
	}

	std::vector<cuda::device::id_t> device_ids(device_count);
	std::iota(device_ids.begin(), device_ids.end(), 0);
	std::remove_if(device_ids.begin(), device_ids.end(),
		[](cuda::device::id_t id) {
			auto properties = cuda::device::get(id).properties();
			return !properties.usable_for_compute()
				or !properties.compute_capability().is_valid();
		}
	);
	if (device_ids.empty()) {
		die_("gpuGetMaxGflopsDeviceId() CUDA error: all devices have compute mode prohibited.");
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
		std::remove_if(device_ids.begin(), device_ids.end(),
			[best_sm_arch](cuda::device::id_t id) {
				return cuda::device::get(id).properties().compute_capability().major < (unsigned) best_sm_arch;
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
inline int findCudaDevice(int argc, const char **argv)
{
	cuda::device::id_t device_id;
	// If the command-line has a device number specified, use it
	if (checkCmdLineFlag(argc, argv, "device"))
	{
		device_id = getCmdLineArgumentInt(argc, argv, "device=");

		if (device_id < 0) { die_("Invalid command line parameter"); }
		else
		{
			device_id = gpuDeviceInit(device_id);
			if (device_id < 0) { die_ ("exiting..."); }
		}
	}
	else
	{
		// Otherwise pick the device with highest Gflops/s
		device_id = gpuGetMaxGflopsDeviceId();
		cuda::device::current::set(device_id);
		auto current_device = cuda::device::current::get();
		std::cout
			<< "GPU Device " << device_id << ": "
			<< "\"" << current_device.name() << "\" "
			<< "with compute capability "
			<< current_device.properties().compute_capability() << "\n";
	}

	return device_id;
}


// General check for CUDA GPU SM Capabilities
inline bool checkCudaCapabilities(int major_version, int minor_version)
{
	auto minimum_cc = cuda::device::make_compute_capability(major_version, minor_version);
	auto device = cuda::device::current::get();
	auto cc = device.properties().compute_capability();
	if (cc >= minimum_cc) {
		std::cout
			<< "  Device " << device.id() << " <" << std::setw(16) << device.properties().name << " >, "
			<< "Compute SM " << cc << " detected\n";
		return true;
	}
	std::cerr << "No GPU device was found that can support CUDA compute capability " << minimum_cc << "\n";
	return false;
}

#endif

// end of CUDA Helper Functions

#endif
