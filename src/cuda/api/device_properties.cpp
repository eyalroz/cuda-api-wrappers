#include <cuda/api/device_properties.hpp>

#include <string>
#include <array>
#include <utility>
#include <algorithm>

// The values hard-coded in this file are based on the information from the following sources:
//
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
// https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html
// https://docs.nvidia.com/cuda/volta-tuning-guide/index.html
// https://docs.nvidia.com/cuda/turing-tuning-guide/index.html
// https://en.wikipedia.org/wiki/CUDA
//
// See specifically the sections on shared memory capacity in the Tuning and Programming
// guides -- it's a somewhat confusing issue.

namespace cuda {
namespace device {

namespace detail {

template <typename T, size_t N>
T get_arch_value(const compute_architecture_t& arch, const std::array<std::pair<unsigned, T>, N>& data)
{
	auto result_iter = std::find_if(
		data.cbegin(), data.cend(),
		[arch](const std::pair<unsigned, T>& pair) {
			return pair.first == arch.major;
		}
	);
	if (result_iter == data.end()) {
		throw std::invalid_argument("No architecture numbered " + std::to_string(arch.major));
	}
	return result_iter->second;
}

template <typename T, typename F, size_t N>
T get_compute_capability_value(
	compute_capability_t compute_capability,
	const std::array<std::pair<unsigned, T>, N>& full_cc_data,
	F architecture_fallback)
{
	auto combined = compute_capability.as_combined_number();
	auto result_iter = std::find_if(
		full_cc_data.cbegin(), full_cc_data.cend(),
		[combined](const std::pair<unsigned, unsigned>& pair) {
			return pair.first == combined;
		}
	);
	if (result_iter != full_cc_data.end()) { return result_iter->second; }
	return (compute_capability.architecture.*architecture_fallback)();
}

} // namespace detail

const char* compute_architecture_t::name(unsigned architecture_number)
{
	static const std::array<std::pair<unsigned, const char*>, 6> arch_names { {
		{ 1, "Tesla"          },
		{ 2, "Fermi"          },
		{ 3, "Kepler"         },
		// Note: No architecture number 4!
		{ 5, "Maxwell"        },
		{ 6, "Pascal"         },
		{ 7, "Volta/Turing"   },
			// Unfortunately, nVIDIA broke with the custom of having the numeric prefix
			// designate the architecture name, with Turing (Compute Capability 7.5 _only_).
	} };
	return detail::get_arch_value(compute_architecture_t{architecture_number}, arch_names);
}

shared_memory_size_t compute_architecture_t::max_shared_memory_per_block() const
{
	enum : shared_memory_size_t { KiB = 1024 };
	// On some architectures, the shared memory / L1 balance is configurable,
	// so you might not get the maxima here without making this configuration
	// setting
	static const std::array<std::pair<unsigned, unsigned>, 6> max_shared_mem_values {
	{
		{ 1,  16 * KiB },
		{ 2,  48 * KiB },
		{ 3,  48 * KiB },
		{ 5,  48 * KiB },
		{ 6,  48 * KiB },
		{ 7,  96 * KiB },
			// this is the Volta figure, Turing is different. Also, values above
			// 48 require a call such as:
			//
			// cudaFuncSetAttribute(
			//     my_kernel,
			//     cudaFuncAttributePreferredSharedMemoryCarveout,
			//     cudaSharedmemCarveoutMaxShared
			// );
			//
			// for details, see the CUDA C Programming Guide at:
			// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
	} };
	return detail::get_arch_value(*this, max_shared_mem_values);
}

unsigned compute_architecture_t::max_resident_warps_per_processor() const {
	static const std::array<std::pair<unsigned, unsigned>, 6> max_resident_warps_values {
	{
		{ 1, 24 },
		{ 2, 48 },
		{ 3, 64 },
		{ 5, 64 },
		{ 6, 64 },
		{ 7, 64 }, // this is the Volta figure, Turing is different
	} };
	return detail::get_arch_value(*this, max_resident_warps_values);
}

unsigned compute_architecture_t::max_warp_schedulings_per_processor_cycle() const {
	static const std::array<std::pair<unsigned, unsigned>, 6>  schedulings_per_cycle_data {
	{
		{ 1, 1 },
		{ 2, 2 },
		{ 3, 4 },
		{ 5, 4 },
		{ 6, 4 },
		{ 7, 4 },
	} };
	return detail::get_arch_value(*this, schedulings_per_cycle_data);
}

unsigned compute_architecture_t::max_in_flight_threads_per_processor() const {
	static const std::array<std::pair<unsigned, unsigned>, 6> max_in_flight_threads_values {
	{
		{ 1,   8 },
		{ 2,  32 },
		{ 3, 192 },
		{ 5, 128 },
		{ 6, 128 },
		{ 7, 128 }, // this is the Volta figure, Turing is different
	} };
	return (detail::get_arch_value(*this, max_in_flight_threads_values));
}

shared_memory_size_t compute_capability_t::max_shared_memory_per_block() const
{
	enum : shared_memory_size_t { KiB = 1024 };

	static const std::array<std::pair<unsigned, unsigned>, 6> max_shared_memory_values {
	{
		{ 37, 112 * KiB },
		{ 75,  64 * KiB },
			// This is the Turing, rather than Volta, figure; but see
			// the note regarding how to actually enable this
	} };
	return
		detail::get_compute_capability_value(
			*this,
			max_shared_memory_values,
			&compute_architecture_t::max_shared_memory_per_block
		);
}

unsigned compute_capability_t::max_resident_warps_per_processor() const {
	static const std::array<std::pair<unsigned, unsigned>, 6> max_resident_warps_values {
	{
		{ 11, 24 },
		{ 12, 32 },
		{ 13, 32 },
		{ 75, 32 },
	} };
	return
		detail::get_compute_capability_value(
			*this,
			max_resident_warps_values,
			&compute_architecture_t::max_resident_warps_per_processor
		);
}

unsigned compute_capability_t::max_warp_schedulings_per_processor_cycle() const {
	static const std::array<std::pair<unsigned, unsigned>, 6> schedulings_per_cycle_values {
	{
		{ 61, 4 },
		{ 62, 4 },
	} };
	return
		detail::get_compute_capability_value(
			*this,
			schedulings_per_cycle_values,
			&compute_architecture_t::max_warp_schedulings_per_processor_cycle
		);
}

unsigned compute_capability_t::max_in_flight_threads_per_processor() const {
	static const std::array<std::pair<unsigned, unsigned>, 6> max_in_flight_threads_values {
	{
		{ 21,  48 },
		{ 60,  64 },
		{ 75,  64 },
	} };
	return
		detail::get_compute_capability_value(
			*this,
			max_in_flight_threads_values,
			&compute_architecture_t::max_in_flight_threads_per_processor
		);
}

} // namespace device
} // namespace cuda
