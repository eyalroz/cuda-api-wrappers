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

namespace std {

std::ostream& operator<<(std::ostream& os, cuda::device::compute_capability_t cc)
{
    return os << cc.major() << '.' << cc.minor();
}

std::ostream& operator<<(std::ostream& os, cuda::multiprocessor_cache_preference_t pref)
{
	return (os << cache_preference_name(pref));
}

std::ostream& operator<<(std::ostream& os, const cuda::device_t& device)
{
	return os << "[id " << device.id() << ']';
}

} // namespace std

[[noreturn]] void die_(const std::string& message)
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

#endif /* EXAMPLES_COMMON_HPP_ */
