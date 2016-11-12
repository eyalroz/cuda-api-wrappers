#include "cuda/api/device_properties.hpp"

#include <string>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <utility>
#include <climits>

namespace cuda {
namespace device {

using std::setw;
using std::left;
using std::setprecision;
using std::setw;
using std::setprecision;

const char* architecture_name(unsigned major_compute_capability_version)
{
	static std::unordered_map<unsigned, std::string> arch_names =
	{
		{ 1u, "Tesla"   },
		{ 2u, "Fermi"   },
		{ 3u, "Kepler"  },
		{ 5u, "Maxwell" },
		{ 6u, "Pascal"  },
	};
	return arch_names.at(major_compute_capability_version).c_str();
}

shared_memory_size_t compute_capability_t::max_shared_memory_per_block() const
{
	using namespace std;
	// On Kepler, you need to actually set the shared memory / L1 balance to get this.
	static unordered_map<unsigned, unsigned> smem_arch_defaults =
	{
		{ 10, 16 * 1024  },
		{ 20, 48 * 1024  },
		{ 30, 48 * 1024  },
		{ 50, 48 * 1024  },
		{ 60, 48 * 1024  },
	};
//	static unordered_map<unsigned, unsigned> smem_by_cc_unique = {	};

	auto cc = as_combined_number();
//	auto it = smem_by_cc_unique.find(cc);
//	if (it != smem_by_cc_unique.end()) { return it->second; }
	return smem_arch_defaults.at(cc - cc % 10);
}

unsigned compute_capability_t::max_resident_warps_per_processor() const {
	using namespace std;
	static unordered_map<unsigned, unsigned> data =
	{
		{ 10, 24 },
		{ 11, 24 },
		{ 12, 32 },
		{ 13, 32 },
		{ 20, 32 },
		{ 30, 48 },
		{ 50, 64 },
		{ 60, 64 },
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
	};
	static unordered_map<unsigned, unsigned> smem_by_cc_unique =
	{
		{ 60, 2 },
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
	static unordered_map<unsigned, unsigned> data =
	{
		{ 10,   8 },
		{ 20,  32 },
		{ 30, 192 },
		{ 50, 128 },
		{ 60, 128 },
	};
	static unordered_map<unsigned, unsigned> data_by_cc_unique =
	{
		{ 21,  48 },
		{ 60,  64 },
	};
	auto cc = as_combined_number();
	auto it = data_by_cc_unique.find(cc);
	if (it != data_by_cc_unique.end()) { return it->second; }
	return data.at(cc - cc % 10);
}

} // namespace device
} // namespace cuda
