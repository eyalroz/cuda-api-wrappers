#include "../../common.hpp"

#include <nvml.h>
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
	auto status = nvmlInit_v2();
	check_status(status);
	unsigned int device_count;
	status = nvmlDeviceGetCount_v2 (&device_count);
	check_status(status);
	std::cout << "Device count: " << device_count << '\n';
	list_sm_clock_rates(device_count);
	nvmlDevice_t device;
	status = nvmlDeviceGetHandleByIndex_v2(0, &device);
	std::cout << "Locking clocks to 500-550\n";
	nvmlDeviceSetGpuLockedClocks(device, 500, 550);
	check_status(status);
	list_sm_clock_rates(device_count);
	std::cout << "Resetting clocks\n";
	nvmlDeviceResetGpuLockedClocks(device);
	check_status(status);
	list_sm_clock_rates(device_count);
}