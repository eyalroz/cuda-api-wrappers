/**
 * @file
 *
 * @brief Common header for CUDA API wrapper example programs which utilize NVRTC
 */
#ifndef EXAMPLES_RTC_COMMON_HPP_
#define EXAMPLES_RTC_COMMON_HPP_

#include <string>
#include <iostream>

#include <cuda/nvrtc.hpp>

#include <iostream>

::std::ostream &operator<<(::std::ostream &os, const cuda::rtc::compilation_options_t &opts)
{
	auto marshalled = marshal(opts);
//	os << '(' << marshalled.option_ptrs().size() << ") compilation options: ";
	bool first_option{true};
	for (auto opt: marshalled.option_ptrs()) {
		if (first_option) { first_option = false; }
		else { os << ' '; }
		os << opt;
	}
	return os;
}

inline void print_compilation_options(cuda::rtc::compilation_options_t compilation_options)
{
	auto marshalled = marshal(compilation_options);
	::std::cout << "Compiling with " << marshalled.option_ptrs().size() << " compilation options:\n";
	for (auto opt: marshalled.option_ptrs()) {
		::std::cout << "Option: " << opt << '\n';
	}
}


#endif // EXAMPLES_RTC_COMMON_HPP_
