/**
 * @file
 *
 * @brief Functions related to mangling & demangling identifiers
 *
 */
#ifndef CUDA_API_WRAPPERS_MANGLING_HPP_
#define CUDA_API_WRAPPERS_MANGLING_HPP_

#if CUDA_VERSION >= 11040
#if !defined(_WIN32) && !defined(WIN32)

#include "detail/span.hpp"
#include "detail/unique_span.hpp"
#include <nv_decode.h>

namespace cuda {

namespace detail_ {

using mangle_status_t = int;

inline void validate_mangling_status(int status)
{
	switch (status) {
	// 0 is fine
	case -1: throw ::std::runtime_error("Memory allocation failure by __cu_demangle for a demangled CUDA identifier");
	case -2: throw ::std::invalid_argument("Mangled identifier passed for demangling was invalid");
	case -3: throw ::std::invalid_argument("Validation of one of the input arguments for a __cu_demangle() call failed");
	}
	return;
}

// TODO: Assuming the length _does_ include the trailing '\0'
inline char* demangle(const char* mangled_identifier, char* buffer, size_t& allocated_size)
{
	int status;
	char* result = __cu_demangle(mangled_identifier, buffer, &allocated_size, &status);
	validate_mangling_status(status);
	return result;
}

inline unique_span<char> demangle(const char* mangled_identifier)
{
	size_t allocated_size { 0 };
	auto demangled = demangle(mangled_identifier, nullptr, allocated_size);
#ifndef NDEBUG
	if (allocated_size <= 1) {
		throw ::std::logic_error("Invalid allocation size returned by __cu_demangle()");
	}
#endif
	return unique_span<char>{demangled, allocated_size - 1, c_free_deleter<char> };
}


} // namespace detail_

inline unique_span<char> demangle(const char* mangled_identifier)
{
	return detail_::demangle(mangled_identifier);
}

template<typename T>
T demangle_as(const char* mangled_identifier)
{
	auto demangled = detail_::demangle(mangled_identifier);
	return { demangled.data(), demangled.data() + demangled.size() };
}

template<>
inline ::std::string demangle_as<::std::string>(const char* mangled_identifier)
{
	auto demangled = detail_::demangle(mangled_identifier);
	return { demangled.data(), demangled.size() };
}

} // namespace cuda

#endif // !defined(_WIN32) && !defined(WIN32)
#endif // CUDA_VERSION >= 11040
#endif // CUDA_API_WRAPPERS_MANGLING_HPP_
