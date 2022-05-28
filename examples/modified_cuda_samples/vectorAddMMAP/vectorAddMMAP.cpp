/**
 * Derived from the nVIDIA CUDA 11.4 samples by
 *
 *   Eyal Rozenberg <eyalroz1@gmx.com>
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 *
 */

/* Vector addition: C = A + B.
 *
 * This sample replaces the device allocation in the vectorAddDrv sample with
 * cuMemMap-ed allocations.  This sample demonstrates that the cuMemMap API
 * allows the user to specify the physical properties of their memory while
 * retaining the contiguous nature of their access, thus not requiring a change
 * in their program structure.
 *
 */

// Includes
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <vector>

#include "../../enumerate.hpp"

#include <cuda/api.hpp>

namespace kernel {

constexpr const char *fatbin_filename = "vectorAddMMAP_kernel.fatbin";
constexpr const char *name = "vectorAdd_kernel";

} // namespace kernel

namespace virtual_mem = cuda::memory::virtual_;
using allocation_properties_t = virtual_mem::physical_allocation::properties_t;
constexpr const auto shared_mem_handle_kind = (virtual_mem::physical_allocation::kind_t)
#if defined(__linux__)
	CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
	CU_MEM_HANDLE_TYPE_WIN32;
#endif
using shared_allocation_handle_t = virtual_mem::physical_allocation::shared_handle_t<shared_mem_handle_kind>;
using mem_region_t = cuda::memory::region_t;
using ::std::vector;

size_t safe_round_up(size_t x, size_t divisor)
{
	// Note: Implementation changed relative to the original sample to
	// reduce the possibility of overflow. Granted, this is bikeshedding;
	// and you might like to throw an exception on overflow anyway
	auto remainder = x % divisor;
	if (remainder == 0) { return x; }
	auto roundup_amount = divisor - remainder;
	if (::std::numeric_limits<size_t>::max() - x < roundup_amount) {
		throw ::std::invalid_argument("A rounding up of " + ::std::to_string(x)
			+ " to a multiple of " + ::std::to_string(divisor) +  " would overflow");
	}
	return x + roundup_amount;
}

size_t div_rounding_up(size_t dividend, size_t divisor)
{
	return dividend/divisor + ((dividend % divisor == 0) ? 0 : 1);
}

// We cannot necessarily satisfy the exact request for a allocated-and-mapped region
// in terms of size, because we want every device to allocate the same amount of memory,
// and also the devices have allocation granularity requirements
cuda::size_t determine_reservation_size(
	cuda::size_t desired_region_size,
	const vector<cuda::device_t> &backing_devices,
	const vector<cuda::device_t> &mapping_devices)
{
	vector<cuda::size_t> allocation_granularities; // for both mapping and backing devices
	cuda::size_t min_overall_granularity{0};

	for (const auto &vec: {backing_devices, mapping_devices} ) {
		for (const auto &device: vec) {
			auto props = virtual_mem::physical_allocation::create_properties_for<shared_mem_handle_kind>(device);
			min_overall_granularity = ::std::max(min_overall_granularity, props.minimum_granularity());
		}
	}

	// Round up the size such that we can evenly split it into a stripe size tha meets the granularity requirements
	auto overall_allocation_size_quantum = backing_devices.size() * min_overall_granularity;
	if (overall_allocation_size_quantum == 0) {
		throw ::std::logic_error("Failed calculating the overall allocation size quantum");
	}
	auto rounded_up_size = safe_round_up(desired_region_size, overall_allocation_size_quantum);
	return rounded_up_size;
}

template <template <typename... Ts> class Container>
struct reserved_range_and_mappings {
	virtual_mem::reserved_address_range_t reserved_range;
	Container<virtual_mem::mapping_t> mappings;
};

/**
 * Allocate virtually contiguous memory backed on separate devices
 *
 * @return a structure containing the "owning" references to the virtual memory address
 * range reservations, and the different mappings from physical allocations to subregions
 * within it - one mapping per device
 *
 * @todo : Do we actually need to keep the reservation handle when a mapping is active?
 *
 * @param backing_devices  Specifies what devices the allocation should be striped across.
 * @param mapping_devices  Specifies what devices need to read/write to the allocation.
 * @param alignment        Additional allignment requirement if desired.
 *
 * @note       The VA mappings will look like the following:
 *
 *     v-stripeSize-v                v-rounding -v
 *     +-----------------------------------------+
 *     |      D1     |      D2     |      D3     |
 *     +-----------------------------------------+
 *     ^-- dptr                      ^-- dptr + size
 *
 * Each device in the residentDevices list will get an equal sized stripe - and some
 * memory in excess of the amount requested may be allocated, to ensure this fact,
 * plus meet the allocation granularity requirements of each device.
 */
reserved_range_and_mappings<::std::vector>
setup_virtual_memory(cuda::size_t desired_region_size,
	const vector<cuda::device_t> &backing_devices,
	const vector<cuda::device_t> &mapping_devices,
	virtual_mem::alignment_t alignment = virtual_mem::alignment::default_)
{
	auto size_to_reserve = determine_reservation_size(desired_region_size, backing_devices, mapping_devices);
	auto stripe_size = size_to_reserve / backing_devices.size();
	auto reserved_range = virtual_mem::reserve(size_to_reserve, alignment);

	vector<virtual_mem::mapping_t> mappings;
	::std::transform(enumerate(backing_devices).cbegin(), enumerate(backing_devices).cend(), ::std::back_inserter(mappings),
		[&](decltype(enumerate(backing_devices))::const_value_type index_and_device) {
			// Note: With C++14, the above statement could use an auto type, simplifying things

			// Note 2: With C++17 we could just have structured bindings in the loop header; instead...
			auto index = index_and_device.index;
			auto device = index_and_device.item;

			// Assign the chunk to the appropriate VA range and release the handle.
			// After mapping the memory, it can be referenced by virtual address.
			// Since we do not need to make any other mappings of this memory or export it,
			// we no longer need and can release the allocation; it will be kept live until
			// it is unmapped.
			auto physical_allocation =
				virtual_mem::physical_allocation::create<shared_mem_handle_kind>(stripe_size, device);
			auto single_device_subregion = reserved_range.region().subregion(stripe_size * index, stripe_size);
			return virtual_mem::map(single_device_subregion, physical_allocation);
		}
	);

	virtual_mem::set_access_mode(reserved_range.region(), mapping_devices, virtual_mem::read_and_write_access);

	return { ::std::move(reserved_range), ::std::move(mappings) };
}

//collect all of the devices whose memory can be mapped from a given device.
vector<cuda::device_t> get_backing_devices(cuda::device_t mapping_device)
{
	vector<cuda::device_t> backing_devices;
	auto devices = cuda::devices();
	::std::copy_if(devices.cbegin(), devices.cend(), ::std::back_inserter(backing_devices),
		[&](cuda::device_t device) {
			return (device == mapping_device) or
				(mapping_device.can_access(device) and device.supports_virtual_memory_management());
		}
	);
	return backing_devices;
}

::std::string get_file_contents(const char *path)
{
	::std::ios::openmode openmode = ::std::ios::in | ::std::ios::binary;
	::std::ifstream ifs(path, openmode);
	if (ifs.bad() or ifs.fail()) {
		throw ::std::system_error(errno, ::std::system_category(), ::std::string("opening ") + path + " in binary read mode");
	}
	::std::ostringstream sstr;
	sstr << ifs.rdbuf();
	return sstr.str();
}

bool results_are_valid(const float *h_A, const float *h_B, const float *h_C, int length)
{
	for (int i = 0; i < length; ++i) {
		float sum = h_A[i] + h_B[i];
		if (::std::fabs(h_C[i] - sum) > 1e-7f) {
			return false;
		}
	}
	return true;
}

int main()
{
	::std::cout << "Vector Addition (using virtual memory mapping)\n";
	int num_elements = 50000;
	size_t size_in_bytes = num_elements * sizeof(float);

	auto device = cuda::device::current::get();

	if (not device.supports_virtual_memory_management()) {
		::std::cout << "Device " << device.id() << " (" << device.name()
				  << ") doesn't support virtual memory management.\n";
		exit(EXIT_SUCCESS);
	}

	//The vector addition happens on cuDevice, so the allocations need to be mapped there.
	vector<cuda::device_t> mapping_devices;
	mapping_devices.push_back(device);

	// Collect devices accessible by the mapping device (cuDevice) into the backing_devices vector.
	vector<cuda::device_t> backing_devices = get_backing_devices(device);
	if (backing_devices.empty()) {
		::std::cout << "No devices can be used for physical allocation for virtual memory mapping" << ::std::endl;
		exit(EXIT_SUCCESS);
	}

	auto fatbin = get_file_contents(kernel::fatbin_filename);
	auto module = cuda::module::create(device, fatbin);
	auto kernel = module.get_kernel(kernel::name);

//	::std::cout << "Kernel \"" << kernel::name << "\" obtained from fatbin file and ready for use." << ::std::endl;

	// TODO: When switching to C++14, replace these with ::std::make_unique
	auto h_A = ::std::unique_ptr<float[]>(new float[num_elements]);
	auto h_B = ::std::unique_ptr<float[]>(new float[num_elements]);
	auto h_C = ::std::unique_ptr<float[]>(new float[num_elements]);

	auto generator = []() { return rand() / (float) RAND_MAX; };
	::std::generate(h_A.get(), h_A.get() + num_elements, generator);
	::std::generate(h_B.get(), h_B.get() + num_elements, generator);

	// Allocate vectors in device memory
	//
	// Note that a call to device::enable_access_to() is not needed even though
	// the backing devices and mapping device are not the same.
	// This is because the cuda::memory::virtual::set_access_mode() call
	// explicitly specifies the cross device mapping.
	// set_access_mode() is still subject to the constraints of cuda::device::peer_to_peer::can_access()
	// for cross device mappings (hence why we filtered the backing devices using device_t::can_access_peer() earlier).
	auto d_A = setup_virtual_memory(size_in_bytes, backing_devices, mapping_devices);
	auto d_B = setup_virtual_memory(size_in_bytes, backing_devices, mapping_devices);
	auto d_C = setup_virtual_memory(size_in_bytes, backing_devices, mapping_devices);

	auto d_A_ptr = d_A.reserved_range.region().start();
	auto d_B_ptr = d_B.reserved_range.region().start();
	auto d_C_ptr = d_C.reserved_range.region().start();

//	::std::cout << "Done setting up virtual memory" << ::std::endl;

	cuda::memory::copy(d_A_ptr, h_A.get(), size_in_bytes);
	cuda::memory::copy(d_B_ptr, h_B.get(), size_in_bytes);

	// Launch the Vector Add CUDA Kernel
	int threads_per_block = 256;
	int blocks_per_grid = div_rounding_up(num_elements, threads_per_block);
	auto launch_config = cuda::make_launch_config(blocks_per_grid, threads_per_block);

	::std::cout << "CUDA kernel launch with " << blocks_per_grid << " blocks of " << threads_per_block << " threads" << ::std::endl;

	cuda::launch(kernel, launch_config,
		d_A_ptr, d_B_ptr, d_C_ptr, num_elements
	);

	cuda::memory::copy(h_C.get(), d_C_ptr, size_in_bytes);

//	::std::cout << "Checking results...\n\n";

	if (results_are_valid(h_A.get(), h_B.get(), h_C.get(), num_elements)) {
		::std::cout << "Test PASSED\n";
		::std::cout << "SUCCESS\n";
	} else {
		::std::cerr << "Result verification FAILED";
		exit(EXIT_FAILURE);
	}
}
