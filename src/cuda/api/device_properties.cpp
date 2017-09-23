#include <cuda/api/device_properties.hpp>

#include <string>
#include <unordered_map>
#include <utility>

namespace cuda {
namespace device {

const char* compute_architecture_t::name(unsigned architecture_number)
{
	static std::unordered_map<unsigned, std::string> arch_names =
	{
		{ 1, "Tesla"   },
		{ 2, "Fermi"   },
		{ 3, "Kepler"  },
		{ 5, "Maxwell" },
		{ 6, "Pascal"  },
		{ 7, "Pascal"  },
	};
	return arch_names.at(architecture_number).c_str();
		// Will throw for invalid architecture numbers!
}

shared_memory_size_t compute_architecture_t::max_shared_memory_per_block() const
{
	enum : shared_memory_size_t { KiB = 1024 };
	// On some architectures, the shared memory / L1 balance is configurable,
	// so you might not get the maxima here without making this configuration
	// setting
	static std::unordered_map<unsigned, unsigned> data =
	{
		{ 1,  16 * KiB },
		{ 2,  48 * KiB },
		{ 3,  48 * KiB },
		{ 5,  64 * KiB },
		{ 6,  64 * KiB },
		{ 7,  96 * KiB },
			// this is a speculative figure based on:
			// https://devblogs.nvidia.com/parallelforall/inside-volta/
	};
	return data.at(major);
}

unsigned compute_architecture_t::max_resident_warps_per_processor() const {
	static std::unordered_map<unsigned, unsigned> data =
	{
		{ 1, 24 },
		{ 2, 48 },
		{ 3, 64 },
		{ 5, 64 },
		{ 6, 64 },
		{ 7, 64 },
	};
	return data.at(major);
}

unsigned compute_architecture_t::max_warp_schedulings_per_processor_cycle() const {
	static std::unordered_map<unsigned, unsigned> data =
	{
		{ 1, 1 },
		{ 2, 2 },
		{ 3, 4 },
		{ 5, 4 },
		{ 6, 2 },
		{ 7, 2 }, // speculation
	};
	return data.at(major);
}

unsigned compute_architecture_t::max_in_flight_threads_per_processor() const {
	static std::unordered_map<unsigned, unsigned> data =
	{
		{ 1,   8 },
		{ 2,  32 },
		{ 3, 192 },
		{ 5, 128 },
		{ 6, 128 },
		{ 7, 128 }, // speculation
	};
	return data.at(major);
}

shared_memory_size_t compute_capability_t::max_shared_memory_per_block() const
{
	enum : shared_memory_size_t { KiB = 1024 };
	static std::unordered_map<unsigned, unsigned> data =
	{
		{ 37, 112 * KiB },
		{ 52,  96 * KiB },
		{ 61,  96 * KiB },
	};
	auto cc = as_combined_number();
	auto it = data.find(cc);
	if (it != data.end()) { return it->second; }
	return architecture().max_shared_memory_per_block();
}

unsigned compute_capability_t::max_resident_warps_per_processor() const {
	static std::unordered_map<unsigned, unsigned> data =
	{
		{ 11, 24 },
		{ 12, 32 },
		{ 13, 32 },
	};
	auto cc = as_combined_number();
	auto it = data.find(cc);
	if (it != data.end()) { return it->second; }
	return architecture().max_resident_warps_per_processor();
}

unsigned compute_capability_t::max_warp_schedulings_per_processor_cycle() const {
	static std::unordered_map<unsigned, unsigned> data =
	{
		{ 61, 4 },
		{ 62, 4 },
	};
	auto cc = as_combined_number();
	auto it = data.find(cc);
	if (it != data.end()) { return it->second; }
	return architecture().max_warp_schedulings_per_processor_cycle();
}

unsigned compute_capability_t::max_in_flight_threads_per_processor() const {
	static std::unordered_map<unsigned, unsigned> data =
	{
		{ 21,  48 },
		{ 60,  64 },
	};
	auto cc = as_combined_number();
	auto it = data.find(cc);
	if (it != data.end()) { return it->second; }
	return architecture().max_in_flight_threads_per_processor();
}

} // namespace device
} // namespace cuda
