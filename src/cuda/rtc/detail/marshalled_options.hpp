#ifndef SRC_CUDA_API_DETAIL_MARSHALLED_OPTIONS_HPP_
#define SRC_CUDA_API_DETAIL_MARSHALLED_OPTIONS_HPP_

#include "../types.hpp"

#include <cstdlib>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>
#include <algorithm>

namespace cuda {

namespace rtc {

namespace detail_ {

// These two structs are streamed to a marshalled options object (see below)
// to indicate the start of a new option or the conclusion of all options,
// respectively

template <typename Delimiter> struct opt_start_t;

template<typename T>
struct is_marshalling_control : ::std::false_type {};

/**
 * This class is necessary for realizing everything we need from
 * the marshalled options: Easy access using an array of pointers,
 * for the C API - and RAIIness for convenience and safety when
 * using the wrapper classes.
 */
class marshalled_options_t {
public:
	using size_type = size_t;

	struct advance_gadget {}; /// triggers an advance() when streamed into an object of this class

	marshalled_options_t()
	{
		option_positions.push_back(0);
	}
	marshalled_options_t(size_type max_num_options)
	{
		option_positions.reserve(max_num_options);
		option_positions.push_back(0);
	}

	marshalled_options_t(const marshalled_options_t&) = delete;
	marshalled_options_t(marshalled_options_t&&) = default;

protected:
	mutable ::std::ostringstream oss;
	mutable ::std::string finalized {};
	::std::vector<size_type> option_positions;
	// Offsets into the eventually-created options string.
	// Note that the last offset is to a not-yet-existing option
public:

	bool empty() const {
		return oss.tellp() == 0;
	}

	template <typename T, typename = ::cuda::detail_::enable_if_t<not detail_::is_marshalling_control<typename ::std::decay<T>::type>::value>>
	marshalled_options_t& operator<<(T&& x)
	{
		oss << x;
		return *this;
	}

	marshalled_options_t& advance()
	{
		oss << '\0';
		option_positions.push_back(oss.tellp());
		return *this;
	}

	::std::vector<const char*> option_ptrs() const {
		finalized = oss.str();
		auto ptrs = ::std::vector<const char*>();
		ptrs.reserve(option_positions.size()-1);
		const char* start = finalized.data();
		::std::transform(option_positions.cbegin(), option_positions.cend() - 1, ::std::back_inserter(ptrs),
			[start] (size_type pos){ return start + pos; });
		return ptrs;
	}
};

inline marshalled_options_t& operator<< (marshalled_options_t& mo, marshalled_options_t::advance_gadget)
{
	mo.advance();
	return mo;
}

template<> struct is_marshalling_control<marshalled_options_t::advance_gadget> : ::std::true_type {};

template<typename Delimiter>
struct is_marshalling_control<opt_start_t<Delimiter>> : ::std::true_type {};

} // namespace detail_

} // namespace rtc

} // namespace cuda

#endif /* SRC_CUDA_API_DETAIL_MARSHALLED_OPTIONS_HPP_ */
