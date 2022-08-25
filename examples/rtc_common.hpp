/**
 * @file
 *
 * @brief Common header for CUDA API wrapper example programs which utilize NVRTC
 */
#ifndef EXAMPLES_RTC_COMMON_HPP_
#define EXAMPLES_RTC_COMMON_HPP_

#include <string>
#include <iostream>

#include <cuda/rtc.hpp>

#include <iostream>

template <typename cuda::source_kind_t Kind>
::std::ostream &operator<<(::std::ostream &os, const cuda::rtc::compilation_options_t<Kind> &opts)
{
	auto marshalled = cuda::rtc::marshal<Kind>(opts);
//	os << '(' << marshalled.option_ptrs().size() << ") compilation options: ";
	bool first_option{true};
	for (auto opt: marshalled.option_ptrs()) {
		if (first_option) { first_option = false; }
		else { os << ' '; }
		os << opt;
	}
	return os;
}

template <cuda::source_kind_t Kind>
inline void print_compilation_options(cuda::rtc::compilation_options_t<Kind> compilation_options)
{
	auto marshalled = cuda::rtc::marshal<Kind>(compilation_options);
	::std::cout << "Compiling with " << marshalled.option_ptrs().size() << " compilation options:\n";
	for (auto opt: marshalled.option_ptrs()) {
		::std::cout << "Option: " << opt << '\n';
	}
}


#endif // EXAMPLES_RTC_COMMON_HPP_
