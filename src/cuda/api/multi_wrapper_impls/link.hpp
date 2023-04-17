//
// Created by eyalroz on 6/10/22.
//

#ifndef CUDA_API_WRAPPERS_LINK_HPP_
#define CUDA_API_WRAPPERS_LINK_HPP_

#include "../link.hpp"
#include "../device.hpp"

inline cuda::device_t link_t::device() const
{
	return cuda::device::get(device_id());
}

inline cuda::context_t link_t::context() const
{
	static constexpr const bool dont_take_ownership { false };
	return context::wrap(device_id(), context_handle_, dont_take_ownership);
}


#endif //CUDA_API_WRAPPERS_LINK_HPP_
