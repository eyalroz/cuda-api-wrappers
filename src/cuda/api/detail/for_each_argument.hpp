#ifndef CUDA_API_WRAPPERS_FOR_EACH_ARGUMENT_HPP
#define CUDA_API_WRAPPERS_FOR_EACH_ARGUMENT_HPP

#include <utility>

namespace cuda {

namespace detail_ {

template <class F>
void for_each_argument(F) { }

template <class F, class... Args>
void for_each_argument(F f, Args&&... args) {
	using arrT = int[];
	static_cast<void>(arrT{(f(::std::forward<Args>(args)), 0)...});
// This:
//	[](...){}((f(::std::forward<Args>(args)), 0)...);
// doesn't guarantee execution order
}

} // namespaced detail_

} // namespaced cuda

#endif //CUDA_API_WRAPPERS_FOR_EACH_ARGUMENT_HPP
