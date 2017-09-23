#include <cuda/api/device_properties.hpp>

#include <string>
#include <unordered_map>
#include <utility>

namespace cuda {
namespace device {

const char* architecture_name(unsigned major_compute_capability_version)
{
	static std::unordered_map<unsigned, std::string> arch_names =
	{
		{ 1u, "Tesla"   },
		{ 2u, "Fermi"   },
		{ 3u, "Kepler"  },
		{ 5u, "Maxwell" },
		{ 6u, "Pascal"  },
		{ 7u, "Pascal"  },
	};
	return arch_names.at(major_compute_capability_version).c_str();
}

shared_memory_size_t compute_capability_t::max_shared_memory_per_block() const
{
	enum : shared_memory_size_t { KiB = 1024 };
	using namespace std;
	// On some architectures, the shared memory / L1 balance is configurable,
	// so you might not get the maxima here without making this configuration
	// setting
	static unordered_map<unsigned, unsigned> architecture_default_shared_memory_sizes =
	{
		{ 10,  16 * KiB },
		{ 20,  48 * KiB },
		{ 30,  48 * KiB },
		{ 50,  64 * KiB },
		{ 60,  64 * KiB },
		{ 70,  96 * KiB },
			// this is a speculative figure based on:
			// https://devblogs.nvidia.com/parallelforall/inside-volta/
	};
	static unordered_map<unsigned, unsigned> shared_memory_sizes_by_compute_capability = {
		{ 37, 112 * KiB },
		{ 52,  96 * KiB },
		{ 61,  96 * KiB },
	};

	auto cc = as_combined_number();
	auto it = shared_memory_sizes_by_compute_capability.find(cc);
	if (it != shared_memory_sizes_by_compute_capability.end()) { return it->second; }
	return architecture_default_shared_memory_sizes.at(cc - cc % 10);
}

unsigned compute_capability_t::max_resident_warps_per_processor() const {
	using namespace std;
	static unordered_map<unsigned, unsigned> data =
	{
		{ 10, 24 },
		{ 11, 24 },
		{ 12, 32 },
		{ 13, 32 },
		{ 20, 48 },
		{ 30, 64 },
		{ 50, 64 },
		{ 60, 64 },
		{ 70, 64 },
	};
	auto cc = as_combined_number();
	auto it = data.find(cc);
	if (it != data.end()) { return it->second; }
	return data.at(cc - cc % 10);

}

unsigned compute_capability_t::max_warp_schedulings_per_processor_cycle() const {
	using namespace std;
	static unordered_map<unsigned, unsigned> data =
	{
		{ 10, 1 },
		{ 20, 2 },
		{ 30, 4 },
		{ 50, 4 },
		{ 60, 2 },
		{ 70, 2 }, // speculation
	};
	static unordered_map<unsigned, unsigned> smem_by_cc_unique =
	{
		{ 61, 4 },
		{ 62, 4 },
	};
	auto cc = as_combined_number();
	auto it = smem_by_cc_unique.find(cc);
	if (it != smem_by_cc_unique.end()) { return it->second; }
	return data.at(cc - cc % 10);
}

unsigned compute_capability_t::max_in_flight_threads_per_processor() const {
	using namespace std;
	static unordered_map<unsigned, unsigned> architecture_defaults =
	{
		{ 10,   8 },
		{ 20,  32 },
		{ 30, 192 },
		{ 50, 128 },
		{ 60, 128 },
		{ 70, 128 }, // speculation
	};
	static unordered_map<unsigned, unsigned> value_by_compute_capability =
	{
		{ 21,  48 },
		{ 60,  64 },
	};
	auto cc = as_combined_number();
	auto it = value_by_compute_capability.find(cc);
	if (it != value_by_compute_capability.end()) { return it->second; }
	return architecture_defaults.at(cc - cc % 10);
}

} // namespace device
} // namespace cuda
