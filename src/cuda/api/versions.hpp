/**
 * @file
 *
 * @brief Wrappers for Runtime API functions involving versions -
 * of the CUDA runtime and of the CUDA driver. Also defines a @ref cuda::version_t
 * class for working with such versions (as they are not really single
 * numbers) - which is what the wrappers return.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_VERSIONS_HPP_
#define CUDA_API_WRAPPERS_VERSIONS_HPP_

#include "error.hpp"

#if CUDA_VERSION >= 12040
#include <nvFatbin.h>
#endif

#include <ostream>
#include <utility>
#include <limits>


namespace cuda {

/**
 * A combination of the major and minor version numbers for a CUDA release
 * into a single integer, e.g. CUDA 11.3 is represented by the combined
 * version number 11300. Se also @ref version_t.
 */
using combined_version_t = int;

/**
 * A structure representing a CUDA release version
 *
 * @note not to be confused with @ref device::compute_capability_t , nor
 * with a CUDA driver version!
 */
struct version_t {
	///@cond
	int major;
	int minor;
	///@endcond

	/// Parse the combined single-number representation, separating it
	static version_t from_single_number(combined_version_t combined_version) noexcept
	{
		return { combined_version / 1000, (combined_version % 100) / 10 };
	}

	///@cond
	operator ::std::pair<int, int>() const noexcept { return { major, minor }; }
	///@endcond
};

///@cond
inline ::std::ostream& operator<<(::std::ostream& os, version_t v)
{
	return os << v.major << '.' << v.minor;
}

// Note: All of comparison operators in this can be made constexpr in C++14

inline bool operator==(const version_t& lhs, const version_t& rhs) noexcept
{
	return lhs.operator ::std::pair<int, int>() == rhs.operator ::std::pair<int, int>();
}

inline bool operator!=(const version_t& lhs, const version_t& rhs) noexcept
{
	return lhs.operator ::std::pair<int, int>() != rhs.operator ::std::pair<int, int>();
}

inline bool operator<(const version_t& lhs, const version_t& rhs) noexcept
{
	return lhs.operator ::std::pair<int, int>() < rhs.operator ::std::pair<int, int>();
}

inline bool operator<=(const version_t& lhs, const version_t& rhs) noexcept
{
	return lhs.operator ::std::pair<int, int>() <= rhs.operator ::std::pair<int, int>();
}

inline bool operator>(const version_t& lhs, const version_t& rhs) noexcept
{
	return lhs.operator ::std::pair<int, int>() > rhs.operator ::std::pair<int, int>();
}

inline bool operator>=(const version_t& lhs, const version_t& rhs) noexcept
{
	return lhs.operator ::std::pair<int, int>() >= rhs.operator ::std::pair<int, int>();
}

// comparison with single integers - as major versions

inline bool operator==(const version_t& lhs, int rhs)  noexcept { return lhs == version_t::from_single_number(rhs); }
inline bool operator!=(const version_t& lhs, int rhs)  noexcept { return lhs != version_t::from_single_number(rhs); }
inline bool operator< (const version_t& lhs, int rhs)  noexcept { return lhs  < version_t::from_single_number(rhs); }
inline bool operator> (const version_t& lhs, int rhs)  noexcept { return lhs  > version_t::from_single_number(rhs); }
inline bool operator<=(const version_t& lhs, int rhs)  noexcept { return lhs <= version_t::from_single_number(rhs); }
inline bool operator>=(const version_t& lhs, int rhs)  noexcept { return lhs >= version_t::from_single_number(rhs); }

///@endcond


namespace version_numbers {

/**
 * This "value" is what the Runtime API returns if no version is
 * supported by the driver
 *
 * @note this is super-ugly, I'd rather n ot use  it at all
 */
constexpr version_t none() noexcept
{
	return { 0, 0 };
}

/**
 * Convert an integer representing a major and minor number (e.g.
 * 55 for major version 5, minor version 5) into the version type
 * we use (@ref version_t).
 */
inline version_t make(combined_version_t combined_version) noexcept
{
	return version_t::from_single_number(combined_version);
}

/**
 * Convert a pair integer representing a major and minor number
 * (e.g. 5 and 5) into the version type we use (@ref version_t).
 */
inline version_t make(int major, int minor) noexcept
{
	return { major, minor };
}

/**
 * Obtains the maximum version of the CUDA Runtime supported by the
 * driver currently loaded by the operating system
 *
 * @todo In future CUDA versions which support C++17 - return
 * an optional
 *
 *
 * @return If an nVIDIA GPU driver is installed on this system,
 * the maximum CUDA version it supports is returned.
 * If no version is supported, @ref none() is returned.
 */
inline version_t corresponding_to_driver() {
	combined_version_t version;
	auto status = cuDriverGetVersion(&version);
	throw_if_error_lazy(status, "Failed obtaining the CUDA driver version");
	return version_t::from_single_number(version);
}

/**
 * Obtains the CUDA Runtime version
 *
 * @note unlike {@ref corresponding_to_driver()}, the value of @ref none() cannot be returned,
 * as we are actually using the runtime to obtain the version.
 */
inline version_t runtime() {
	combined_version_t version;
	auto status = cudaRuntimeGetVersion(&version);
	throw_if_error_lazy(status, "Failed obtaining the CUDA runtime version");
	return version_t::from_single_number(version);
}

#if CUDA_VERSION >= 12040

inline version_t fatbin() {
	unsigned int major { 0 }, minor { 0 };

	auto status = nvFatbinVersion(&major, &minor);
	throw_if_error_lazy(status, "Failed obtaining the nvfatbin library version");
#ifndef NDEBUG
	if ((major == 0) or (major > ::std::numeric_limits<int>::max())
		or (minor == 0) or (minor > ::std::numeric_limits<int>::max())) {
		throw ::std::logic_error("Invalid version encountered: ("
			+ ::std::to_string(major) + ", " + ::std::to_string(minor) + ')' );
	}
#endif
	return version_t{ static_cast<int>(major), static_cast<int>(minor) };
}
#endif // CUDA_VERSION >= 12040

} // namespace version_numbers
} // namespace cuda

#endif // CUDA_API_WRAPPERS_VERSIONS_HPP_
