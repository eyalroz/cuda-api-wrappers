#pragma once
#ifndef CUDA_API_WRAPPERS_ERROR_HPP_
#define CUDA_API_WRAPPERS_ERROR_HPP_

#include "cuda/api/types.h"

#ifdef HAVE_KERNEL_TESTER_UTILS
// This wrapper exception will include a stack trace;
// also, it can take multiple arguments in the ctor which will
// be piped into a stringstream; but we won't make use of that here
// to ensure compatibility with the single-string ctor of an
// std::runtime_error
#include "util/exception.h"
using cuda_inspecific_runtime_error = util::runtime_error;
#else
#include <stdexcept>
using cuda_inspecific_runtime_error = std::runtime_error;
#endif

#include <cuda_runtime_api.h>
#include <type_traits>
#include <string>

namespace cuda {

inline bool is_success(status_t status)  { return status == cudaSuccess; }
inline bool is_failure(status_t status)  { return status != cudaSuccess; }

namespace detail {

template <typename I, bool UpperCase = false>
std::string as_hex(I x, unsigned hex_string_length = 2*sizeof(I) )
{
	static_assert(std::is_unsigned<I>::value, "only signed representations are supported");
	enum { bits_per_hex_digit = 4 }; // = log_2 of 16
	static const char* digit_characters =
		UpperCase ? "0123456789ABCDEF" : "0123456789abcdef" ;

	std::string result(hex_string_length,'0');
	for (unsigned digit_index = 0; digit_index < hex_string_length ; digit_index++)
	{
		size_t bit_offset = (hex_string_length - 1 - digit_index) * bits_per_hex_digit;
		auto hexadecimal_digit = (x >> bit_offset) & 0xF;
		result[digit_index] = digit_characters[hexadecimal_digit];
	}
    return result;
}

// TODO: Perhaps find a way to avoid the extra function, so that as_hex() can
// be called for pointer types as well? Would be easier with boost's uint<T>...
template <typename I, bool UpperCase = false>
inline std::string ptr_as_hex(const I* ptr, unsigned hex_string_length = 2*sizeof(I*) )
{
	return as_hex((size_t) ptr, hex_string_length);
}




} // namespace detail

/**
 * A (base?) class for exceptions raised by CUDA code
 */
class runtime_error : public cuda_inspecific_runtime_error {
public:
	// TODO: Constructor chaining; and perhaps allow for more construction mechanisms?
	runtime_error(cuda::status_t error_code) :
		cuda_inspecific_runtime_error(cudaGetErrorString(error_code)),
		error_code_(error_code)
	{ }
	// I wonder if I should do this the other way around
	runtime_error(cuda::status_t error_code, const std::string& what_arg) :
		cuda_inspecific_runtime_error(what_arg + ": " + cudaGetErrorString(error_code)),
		error_code_(error_code)
	{ }

	status_t error_code() const { return error_code_; }

private:
	status_t error_code_;
};

// TODO: The following could use std::optiomal arguments - which would
// prevent the need for dual versions of the functions - but we're
// not writing C++17 here

inline void throw_if_error(
	cuda::status_t error_code, std::string message) noexcept(false)
{
	if (is_failure(error_code)) { throw runtime_error(error_code, message); }
}

inline void throw_if_error(cuda::status_t error_code) noexcept(false)
{
	if (is_failure(error_code)) { throw runtime_error(error_code); }
}

namespace errors {
enum : bool {
	dont_clear = false,
	clear = true
};
} // namespace errors

inline void ensure_no_outstanding_error(
	std::string message, bool clear_any_error = errors::clear) noexcept(false)
{
	auto last_status = clear_any_error ? cudaGetLastError() : cudaPeekAtLastError();
	throw_if_error(last_status, message);
}

inline void ensure_no_outstanding_error(bool clear_any_error = errors::clear) noexcept(false)
{
	auto last_status = clear_any_error ? cudaGetLastError() : cudaPeekAtLastError();
	throw_if_error(last_status);
}

/**
 * Reset the CUDA status to cudaSuccess.
 */
inline void clear_status() { cudaPeekAtLastError(); }



} // namespace cuda

#endif /* CUDA_API_WRAPPERS_ERROR_HPP_ */
