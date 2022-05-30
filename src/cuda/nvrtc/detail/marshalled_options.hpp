#ifndef SRC_CUDA_API_DETAIL_MARSHALLED_OPTIONS_HPP_
#define SRC_CUDA_API_DETAIL_MARSHALLED_OPTIONS_HPP_

#include <cuda/nvrtc/types.hpp>

#include <cstdlib>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>

namespace cuda {

namespace rtc {

/**
 * This class is necessary for realizing everything we need from
 * the marshalled options: Easy access using an array of pointers,
 * for the C API - and RAIIness for convenience and safety when
 * using the wrapper classes.
 */
class marshalled_options_t {
public:
	using size_type = size_t;

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
	::std::ostringstream oss;
	mutable ::std::string finalized {};
	::std::vector<size_type> option_positions;
		// Offsets into the eventually-created options string.
		// Note that the last offset is to a not-yet-existing option
public:

	template <typename T>
	marshalled_options_t& operator<<(T&& x)
	{
		oss << x;
		return *this;
	}

	// TODO: Make a similar "stream manipulator"?
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

namespace detail_ {

void optend() { }

} // namespace detail_

marshalled_options_t& operator<< (marshalled_options_t& mo, decltype(detail_::optend))
{
	mo.advance();
	return mo;
}

} // namespace rtc

} // namespace cuda

#endif /* SRC_CUDA_API_DETAIL_MARSHALLED_OPTIONS_HPP_ */
