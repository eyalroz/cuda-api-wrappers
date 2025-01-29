/**
 * @file
 *
 * @brief Implementation of methods and helper functions for device-property-related classes.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_DETAIL_DEVICE_PROPERTIES_HPP_
#define CUDA_API_WRAPPERS_DETAIL_DEVICE_PROPERTIES_HPP_

#include "../device_properties.hpp"

///@cond

namespace cuda {
namespace device {

// compute_architecture_t-related functions

inline constexpr bool compute_architecture_t::is_valid() const noexcept
{
	return (major > 0) and (major < 9999); // Picked this up from somewhere in the CUDA code
}

inline constexpr bool operator ==(const compute_architecture_t& lhs, const compute_architecture_t& rhs) noexcept
{
	return lhs.major == rhs.major;
}
inline constexpr bool operator !=(const compute_architecture_t& lhs, const compute_architecture_t& rhs) noexcept
{
	return lhs.major != rhs.major;
}
inline constexpr bool operator <(const compute_architecture_t& lhs, const compute_architecture_t& rhs) noexcept
{
	return lhs.major < rhs.major;
}
inline constexpr bool operator <=(const compute_architecture_t& lhs, const compute_architecture_t& rhs) noexcept
{
	return lhs.major < rhs.major;
}
inline constexpr bool operator >(const compute_architecture_t& lhs, const compute_architecture_t& rhs) noexcept
{
	return lhs.major > rhs.major;
}
inline constexpr bool operator >=(const compute_architecture_t& lhs, const compute_architecture_t& rhs) noexcept
{
	return lhs.major > rhs.major;
}

namespace detail_ {

constexpr const int invalid_compute_capability_return { 0 };
enum : memory::shared::size_t { KiB = 1024 };

template <typename T>
inline T ensure_cc_attribute_validity(T v, const compute_capability_t& cc)
{
	if (v == detail_::invalid_compute_capability_return) {
		throw ::std::invalid_argument("Compute capability unknown: " + ::std::to_string(cc.as_combined_number()));
	}
	return v;
}

template <>
inline const char* ensure_cc_attribute_validity<const char*>(const char* v, const compute_capability_t& cc)
{
	if (v == nullptr) {
		throw ::std::invalid_argument("Compute capability unknown: " + ::std::to_string(cc.as_combined_number()));
	}
	return v;
}

inline constexpr const char* architecture_name(const compute_architecture_t& arch)
{
	return
		(arch.major ==  1) ? "Tesla" :
		(arch.major ==  2) ? "Fermi" :
		(arch.major ==  3) ? "Kepler" :
			// Note: No architecture number 4!
		(arch.major ==  5) ? "Maxwell" :
		(arch.major ==  6) ? "Pascal" :
		(arch.major ==  7) ? "Volta/Turing" :
			// Unfortunately, nVIDIA broke with the custom of having the numeric prefix
			// designate the architecture name, with Turing (Compute Capability 7.5 _only_).
		(arch.major ==  8) ? "Ampere/Lovelace" :
        (arch.major ==  9) ? "Hopper" :
        (arch.major == 10) ? "Blackwell" :
            // Note: No architecture number 11!
        (arch.major == 12) ? "Blackwell" :
            // Note: As of 2025-01-28, NVIDIA claims both architectures 10 and 12 would
            // be named "Blackwell".
		nullptr;
}

} // namespace detail_

inline const char* compute_architecture_t::name() const {
	auto name_ = detail_::architecture_name(*this);
	if (name_ == nullptr) {
		throw ::std::invalid_argument("No known architecture numbered " + ::std::to_string(major));
	}
	return name_;
}

// compute_capability_t-related

inline constexpr bool operator ==(const compute_capability_t& lhs, const compute_capability_t& rhs) noexcept
{
	return lhs.major() == rhs.major() and lhs.minor_ == rhs.minor_;
}
inline constexpr bool operator !=(const compute_capability_t& lhs, const compute_capability_t& rhs) noexcept
{
	return lhs.major() != rhs.major() or lhs.minor_ != rhs.minor_;
}
inline constexpr bool operator <(const compute_capability_t& lhs, const compute_capability_t& rhs) noexcept
{
	return lhs.major() < rhs.major() or (lhs.major() == rhs.major() and lhs.minor_ < rhs.minor_);
}
inline constexpr bool operator <=(const compute_capability_t& lhs, const compute_capability_t& rhs) noexcept
{
	return lhs.major() < rhs.major() or (lhs.major() == rhs.major() and lhs.minor_ <= rhs.minor_);
}
inline constexpr bool operator >(const compute_capability_t& lhs, const compute_capability_t& rhs) noexcept
{
	return lhs.major() > rhs.major() or (lhs.major() == rhs.major() and lhs.minor_ > rhs.minor_);
}
inline constexpr bool operator >=(const compute_capability_t& lhs, const compute_capability_t& rhs) noexcept
{
	return lhs.major() > rhs.major() or (lhs.major() == rhs.major() and lhs.minor_ >= rhs.minor_);
}

inline constexpr bool compute_capability_t::is_valid() const noexcept
{
	return architecture.is_valid() and (minor_ > 0) and (minor_ < 9999);
		// Picked this up from the CUDA code somewhere
}

inline constexpr compute_capability_t compute_capability_t::from_combined_number(unsigned combined) noexcept
{
	return  compute_capability_t{ { combined / 10 }, combined % 10 };
}

inline constexpr unsigned compute_capability_t::as_combined_number() const noexcept { return major() * 10 + minor_; }


inline constexpr compute_capability_t make_compute_capability(unsigned combined) noexcept
{
	return compute_capability_t::from_combined_number(combined);
}

inline constexpr compute_capability_t make_compute_capability(unsigned major, unsigned minor) noexcept
{
	return { {major}, minor };
}

namespace detail_ {

// Based on `_ConvertSMVer2Cores()` from the CUDA samples
inline constexpr unsigned max_in_flight_threads_per_processor(const compute_capability_t& cc)
{
	return
		(cc.architecture.major == 1)    ?   8 :
		(cc.architecture.major == 2)    ?  32 :
		(cc.architecture.major == 3)    ? 192 :
		// Note: No architecture number 4!
		(cc.architecture.major == 5)    ? 128 :
		(cc.as_combined_number() == 60) ?  64 :
		(cc.architecture.major == 6)    ? 128 :
		(cc.architecture.major == 7)    ?  64 :
		(cc.as_combined_number() == 80) ?  64 :
		(cc.architecture.major == 8)    ? 128 :
		(cc.architecture.major == 9)    ? 128 :
		invalid_compute_capability_return;
}

inline constexpr unsigned max_warp_schedulings_per_processor_cycle(const compute_capability_t& cc)
{
	return
		(cc.architecture.major == 1)    ?  1 :
		(cc.architecture.major == 2)    ?  2 :
		(cc.architecture.major == 3)    ?  4 :
		// Note: No architecture number 4!
		(cc.architecture.major == 5)    ?  4 :
		(cc.as_combined_number() == 60) ?  2 :
		(cc.architecture.major == 6)    ?  4 :
		(cc.architecture.major == 7)    ?  4 :
		(cc.architecture.major == 8)    ?  4 :
		(cc.architecture.major == 9)    ?  4 :
		invalid_compute_capability_return;
}

/**
 * @note this is not the maximum amount per preprocessor, only per block!
 *
 * @note Remember that regardless of the value you get from this function,
 * to use more than 48 KiB per block you may need a call such as:
 *
 *	 cudaFuncSetAttribute(
 *	     my_kernel,
 *	     cudaFuncAttributePreferredSharedMemoryCarveout,
 *	     cudaSharedmemCarveoutMaxShared
 *	 );
 *
 * for details, see the CUDA Programming Guide, section K.7.3
 */
inline constexpr unsigned max_shared_memory_per_block(const compute_capability_t& cc)
{
	return
		(cc.architecture.major ==  1)    ?  16 * KiB :
		(cc.architecture.major ==  2)    ?  48 * KiB :
		(cc.architecture.major ==  3)    ?  48 * KiB :
		// Note: No architecture number 4!
		(cc.architecture.major ==  5)    ?  48 * KiB :
		(cc.architecture.major ==  6)    ?  48 * KiB :
		(cc.as_combined_number() ==   7)  ?  64 * KiB : // of 128
		(cc.as_combined_number() ==  72) ?  48 * KiB : // of 128
		(cc.as_combined_number() ==  75) ?  64 * KiB : // of  96
		(cc.architecture.major ==  7)    ?  96 * KiB : // of 128
		(cc.as_combined_number() ==  80) ? 163 * KiB : // of 192
		(cc.as_combined_number() ==  86) ?  99 * KiB : // of 128
		(cc.as_combined_number() ==  87) ? 163 * KiB : // of 192
		(cc.as_combined_number() ==  89) ?  99 * KiB : // of 100
        (cc.as_combined_number() ==  90) ? 227 * KiB : // of 256
        (cc.architecture.major == 10) ? 99 * KiB : // of 256
        (cc.architecture.major == 12) ? 99 * KiB : // of 256
		invalid_compute_capability_return;
}


inline constexpr unsigned max_resident_warps_per_processor(const compute_capability_t& cc) noexcept
{
	return
		(cc.architecture.major == 1)    ?  24 :
		(cc.architecture.major == 2)    ?  48 :
		(cc.architecture.major == 3)    ?  64 :
		// Note: No architecture number 4!
		(cc.architecture.major == 5)    ?  64 :
		(cc.architecture.major == 6)    ?  64 :
		(cc.as_combined_number() == 75) ?  32 :
		(cc.architecture.major == 7)    ?  64 :
		(cc.as_combined_number() == 80) ?  64 :
		(cc.architecture.major == 8)    ?  48 :
		(cc.as_combined_number() == 90) ?  64 :
		invalid_compute_capability_return;
}

} // namespace detail_

inline unsigned compute_capability_t::max_warp_schedulings_per_processor_cycle() const
{
	return detail_::ensure_cc_attribute_validity(detail_::max_warp_schedulings_per_processor_cycle(*this), *this);
}

inline unsigned compute_capability_t::max_in_flight_threads_per_processor() const
{
	return detail_::ensure_cc_attribute_validity(detail_::max_in_flight_threads_per_processor(*this), *this);
}

inline unsigned compute_capability_t::max_shared_memory_per_block() const
{
	return detail_::ensure_cc_attribute_validity(detail_::max_shared_memory_per_block(*this), *this);
}

inline unsigned compute_capability_t::max_resident_warps_per_processor() const
{
	return detail_::ensure_cc_attribute_validity(detail_::max_resident_warps_per_processor(*this), *this);
}

// properties_t-related

inline bool properties_t::usable_for_compute() const noexcept
{
	return computeMode != cudaComputeModeProhibited;
}

} // namespace device
} // namespace cuda

namespace std {

  template <>
  struct hash<cuda::device::compute_capability_t>
  {
	::std::size_t operator()(const cuda::device::compute_capability_t& cc) const noexcept
	{
	  using ::std::hash;

	  // Compute individual hash values for first,
	  // second and third and combine them using XOR
	  // and bit shifting:

	  return hash<unsigned>()(cc.major()) ^ (hash<unsigned>()(cc.minor()) << 1);
	}
  };

} // namespace std

///@endcond

#endif // CUDA_API_WRAPPERS_DETAIL_DEVICE_PROPERTIES_HPP_
