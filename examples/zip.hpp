/**
 * @file
 *
 * @brief A function for applying other functions to corresponding elements
 * in multiple iterable sequences or containers. Small C++11 implementation not
 * requiring Boost (which offers boost::combine() and boost::zip_iterator.
 *
 * @todo Use a tag so that `for_each_zipped()` can be used for containers as well.
 *
 * Example:

int main () {
	::std::vector<int> v1{1,2,3};
	::std::vector<int> v2{3,2,1};
	::std::vector<float> v3{1.2,2.4,9.0};
	::std::vector<float> v4{1.2,2.4,9.0};
	::std::cout << "Using zipped iterators:\n";
	for_each_zipped([](int i,int j,float k,float l) {
			::std::cout << i << " " << j << " " << k << " " << l << '\n';
		},
		v1.begin(),v1.end(),v2.begin(),v3.begin(),v4.begin());
	::std::cout << "\nUsing zipped containers:\n";
	for_each_zipped_containers([](int i,int j,float k,float l) {
			::std::cout << i << " " << j << " " << k << " " << l << '\n';
		},
		v1, v2, v3, v4);
}

 */
#ifndef ZIP_FOREACH_HPP_
#define ZIP_FOREACH_HPP_

#include <vector>
#include <iostream>

namespace detail {

struct advance {
	template <typename T> void operator()(T& t) const { ++t; }
};

// Adaptation of for_each_arg, see:
// https://isocpp.org/blog/2015/01/for-each-argument-sean-parent
template <class... Iterators>
void advance_all(Iterators&... iterators) {
	[](...){}((advance{}(iterators), 0)...);
}

} // namespace detail

template <typename F, typename Iterator, typename ... ExtraIterators>
F for_each_zipped(F func, Iterator begin, Iterator end, ExtraIterators ... extra_iterators)
{
	for(;begin != end; ++begin, detail::advance_all(extra_iterators...))
		func(*begin, *(extra_iterators)... );
	return func;
}
template <typename F, typename Container, typename... ExtraContainers>
F for_each_zipped_containers(F func, Container& container, ExtraContainers& ... extra_containers)
{
	return for_each_zipped(func, ::std::begin(container), ::std::end(container), ::std::begin(extra_containers)...);
}


#endif /*  ZIP_FOREACH_HPP_ */
