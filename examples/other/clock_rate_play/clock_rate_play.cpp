#include "../../common.hpp"

#include <cuda/nvml.hpp>
#include <iostream>
#include <cstdlib>

void check_status(nvmlReturn_t status)
{
	if (status != NVML_SUCCESS) {
		std::cerr << "NVML error: " << nvmlErrorString(status);
		exit(EXIT_FAILURE);
	}
}

void list_sm_clock_rates(unsigned int device_count)
{
	nvmlReturn_t status;
	for(int device_id = 0; device_id < (int) device_count; device_id++) {
		nvmlDevice_t device;
		status = nvmlDeviceGetHandleByIndex_v2(0, &device);
		// nvmlClockType_t clock_type { NVML_CLOCK_SM };
		check_status(status);
		unsigned int sm_clock_info;
		status = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_clock_info);
		check_status(status);
		unsigned int sm_max_clock_info;
		status = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &sm_max_clock_info);
		check_status(status);
		std::cout << "Device " << device_id << " SM clock is: " << std::setw(4) << sm_clock_info
			<< " MHz (with maximum possible " << std::setw(4) << sm_max_clock_info << " MHz)\n";
	}
}

int main()
{
	cuda::nvml::initialize();
	auto device_count = cuda::nvml::device::count();
	std::cout << "Device count: " << device_count << '\n';
	if (device_count == 0) { exit(EXIT_SUCCESS); }
	list_sm_clock_rates(device_count);
	auto cuda_device = cuda::device::default_();
	// auto nvml_device = cuda::nvml::device::detail_::get_handle(0);
	std::cout << "Locking clocks to 500-550\n";
	cuda_device.set_clock_globally(cuda::device_t::clocked_entity_t::sm, {500, 550});
	//check_status(status);
	list_sm_clock_rates(device_count);
	std::cout << "Resetting clocks\n";
	cuda_device.reset_clock(cuda::device_t::clocked_entity_t::sm);
	list_sm_clock_rates(device_count);
}