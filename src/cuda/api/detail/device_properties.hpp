/**
 * @file detail_/device_properties.hpp
 *
 * @brief Implementation of methods and helper functions for device-property-related classes.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_DETAIL_DEVICE_PROPERTIES_HPP_
#define CUDA_API_WRAPPERS_DETAIL_DEVICE_PROPERTIES_HPP_

#include <cuda/api/device_properties.hpp>

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

constexpr const int invalid_architecture_return { 0 };
enum : memory::shared::size_t { KiB = 1024 };

template <typename T>
inline T ensure_arch_property_validity(T v, const compute_architecture_t& arch)
{
	if (v == detail_::invalid_architecture_return) {
		throw ::std::invalid_argument("No architecture numbered " + ::std::to_string(arch.major));
	}
	return v;
}

template <>
inline const char* ensure_arch_property_validity<const char*>(const char* v, const compute_architecture_t& arch)
{
	if (v == nullptr) {
		throw ::std::invalid_argument("No architecture numbered " + ::std::to_string(arch.major));
	}
	return v;
}

inline constexpr const char* architecture_name(const compute_architecture_t& arch)
{
	return
		(arch.major == 1) ? "Tesla" :
		(arch.major == 2) ? "Fermi" :
		(arch.major == 3) ? "Kepler" :
			// Note: No architecture number 4!
		(arch.major == 5) ? "Maxwell" :
		(arch.major == 6) ? "Pascal" :
		(arch.major == 7) ? "Volta/Turing" :
			// Unfortunately, nVIDIA broke with the custom of having the numeric prefix
			// designate the architecture name, with Turing (Compute Capability 7.5 _only_).
		(arch.major == 8) ? "Ampere" :
		nullptr;
}

inline constexpr memory::shared::size_t max_shared_memory_per_block(const compute_architecture_t& arch)
{
	return
		(arch.major == 1) ?  16 * KiB :
		(arch.major == 2) ?  48 * KiB :
		(arch.major == 3) ?  48 * KiB :
		// Note: No architecture number 4!
		(arch.major == 5) ?  48 * KiB :
		(arch.major == 6) ?  48 * KiB :
		(arch.major == 7) ?  96 * KiB :
			// this is the Volta figure, Turing is different. Also, values above 48 require a call such as:
			//
			// cudaFuncSetAttribute(
			//     my_kernel,
			//     cudaFuncAttributePreferredSharedMemoryCarveout,
			//     cudaSharedmemCarveoutMaxShared
			// );
			//
			// for details, see the CUDA C Programming Guide.
		invalid_architecture_return;
}

inline unsigned max_resident_warps_per_processor(const compute_architecture_t& arch)
{
	return
		(arch.major == 1) ?  24 :
		(arch.major == 2) ?  48 :
		(arch.major == 3) ?  64 :
		// Note: No architecture number 4!
		(arch.major == 5) ?  64 :
		(arch.major == 6) ?  64 :
		(arch.major == 7) ?  64 : // this is the Volta figure, Turing is different
		invalid_architecture_return;
}

inline unsigned max_warp_schedulings_per_processor_cycle(const compute_architecture_t& arch)
{
	return
		(arch.major == 1) ?  1 :
		(arch.major == 2) ?  2 :
		(arch.major == 3) ?  4 :
		// Note: No architecture number 4!
		(arch.major == 5) ?  4 :
		(arch.major == 6) ?  4 :
		(arch.major == 7) ?  4 :
		invalid_architecture_return;
}

inline constexpr unsigned max_in_flight_threads_per_processor(const compute_architecture_t& arch)
{
	return
		(arch.major == 1) ?   8 :
		(arch.major == 2) ?  32 :
		(arch.major == 3) ? 192 :
		// Note: No architecture number 4!
		(arch.major == 5) ? 128 :
		(arch.major == 6) ? 128 :
		(arch.major == 7) ? 128 : // this is the Volta figure, Turing is different
		invalid_architecture_return;
}

} // namespace detail_

inline const char* compute_architecture_t::name() const {
	return detail_::ensure_arch_property_validity(detail_::architecture_name(*this), *this);
}

inline unsigned compute_architecture_t::max_in_flight_threads_per_processor() const
{
	return detail_::ensure_arch_property_validity(detail_::max_in_flight_threads_per_processor(*this), *this);
}

inline unsigned compute_architecture_t::max_shared_memory_per_block() const
{
	return detail_::ensure_arch_property_validity(detail_::max_shared_memory_per_block(*this), *this);
}

inline unsigned compute_architecture_t::max_resident_warps_per_processor() const
{
	return detail_::ensure_arch_property_validity(detail_::max_resident_warps_per_processor(*this), *this);
}

inline unsigned compute_architecture_t::max_warp_schedulings_per_processor_cycle() const
{
	return detail_::ensure_arch_property_validity(detail_::max_warp_schedulings_per_processor_cycle(*this), *this);
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
		// Picked this up from the CUDA code somwhere
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

inline constexpr unsigned max_in_flight_threads_per_processor(const compute_capability_t& cc)
{
	return
		cc.as_combined_number() == 21 ? 48 :
		cc.as_combined_number() == 60 ? 64 :
		cc.as_combined_number() == 75 ? 64 :
		max_in_flight_threads_per_processor(cc.architecture);
}

inline constexpr unsigned max_warp_schedulings_per_processor_cycle(const compute_capability_t& cc)
{
	return
		cc.as_combined_number() == 61 ? 4 :
		cc.as_combined_number() == 62 ? 4 :
		max_warp_schedulings_per_processor_cycle(cc.architecture);
}

inline constexpr unsigned max_shared_memory_per_block(const compute_capability_t& cc)
{
	return
		cc.as_combined_number() == 37 ? 112 * KiB :
		cc.as_combined_number() == 75 ?  64 * KiB :
		max_shared_memory_per_block(cc.architecture);
}


inline constexpr unsigned max_resident_warps_per_processor(const compute_capability_t& cc)
{
	return
		cc.as_combined_number() == 11 ? 24 * KiB :
		cc.as_combined_number() == 12 ? 32 * KiB :
		cc.as_combined_number() == 13 ? 32 * KiB :
		cc.as_combined_number() == 75 ? 32 * KiB :
		max_resident_warps_per_processor(cc.architecture);
}

} // namespace detail_

inline unsigned compute_capability_t::max_in_flight_threads_per_processor() const
{
	return detail_::ensure_arch_property_validity(detail_::max_in_flight_threads_per_processor(*this), architecture);
}

inline unsigned compute_capability_t::max_warp_schedulings_per_processor_cycle() const
{
	return detail_::ensure_arch_property_validity(detail_::max_warp_schedulings_per_processor_cycle(*this), architecture);
}

inline unsigned compute_capability_t::max_shared_memory_per_block() const
{
	return detail_::ensure_arch_property_validity(detail_::max_shared_memory_per_block(*this), architecture);
}

inline unsigned compute_capability_t::max_resident_warps_per_processor() const
{
	return detail_::ensure_arch_property_validity(detail_::max_resident_warps_per_processor(*this), architecture);
}

// properties_t-related

inline bool properties_t::usable_for_compute() const noexcept
{
	return computeMode != cudaComputeModeProhibited;
}


} // namespace device
} // namespace cuda

///@endcond

#endif // CUDA_API_WRAPPERS_DETAIL_DEVICE_PROPERTIES_HPP_
