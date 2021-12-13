/**
 * @file examples/common.hpp
 *
 * @brief Common header for many/most/all CUDA API wrapper example programs.
 */
#ifndef EXAMPLES_COMMON_HPP_
#define EXAMPLES_COMMON_HPP_

#include <cuda/runtime_api.hpp>

#include <string>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <cstring>
#include <system_error>
#include <memory>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include <iomanip>
#include <numeric>

const char* cache_preference_name(cuda::multiprocessor_cache_preference_t pref)
{
	static const char* cache_preference_names[] = {
		"No preference",
		"Equal L1 and shared memory",
		"Prefer shared memory over L1",
		"Prefer L1 over shared memory",
	};
	return cache_preference_names[(off_t) pref];
}

const char* host_thread_synch_scheduling_policy_name(cuda::host_thread_synch_scheduling_policy_t policy)
{
	static const char *names[] = {
		"heuristic",
		"spin",
		"yield",
		"INVALID",
		"block",
		nullptr
	};
	return names[(off_t) policy];
}


namespace std {

std::ostream& operator<<(std::ostream& os, cuda::device::compute_capability_t cc)
{
    return os << cc.major() << '.' << cc.minor();
}

std::ostream& operator<<(std::ostream& os, cuda::multiprocessor_cache_preference_t pref)
{
	return (os << cache_preference_name(pref));
}

std::ostream& operator<<(std::ostream& os, cuda::host_thread_synch_scheduling_policy_t pref)
{
	return (os << host_thread_synch_scheduling_policy_name(pref));
}

std::ostream& operator<<(std::ostream& os, const cuda::device_t& device)
{
	return os << cuda::device::detail_::identify(device.id());
}

std::ostream& operator<<(std::ostream& os, const cuda::stream_t& stream)
{
	return os << cuda::stream::detail_::identify(stream.handle(), stream.device().id());
}

} // namespace std

[[noreturn]] bool die_(const std::string& message)
{
	std::cerr << message << "\n";
	exit(EXIT_FAILURE);
}

#define assert_(cond) \
{ \
	auto evaluation_result = (cond); \
	if (not evaluation_result) \
		die_("Assertion failed at line " + std::to_string(__LINE__) + ": " #cond); \
}

// Note: This will only work correctly for positive values
template <typename U1, typename U2>
typename std::common_type<U1,U2>::type div_rounding_up(U1 dividend, U2 divisor)
{
	return dividend / divisor + !!(dividend % divisor);
}

#endif // EXAMPLES_COMMON_HPP_
