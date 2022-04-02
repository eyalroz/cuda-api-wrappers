/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard modules. Specifically:
 *
 * 1. Functions in the `cuda::module` namespace.
 * 2. Methods of @ref `cuda::module_t` and possibly some relates classes.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_MODULE_HPP_
#define MULTI_WRAPPER_IMPLS_MODULE_HPP_

#include "../device.hpp"
#include "../context.hpp"
#include "../module.hpp"

namespace cuda {

namespace module {

namespace detail_{

inline device::primary_context_t get_context_for(device_t& locus) { return locus.primary_context(); }

} // namespace detail_

} // namespace module

inline context_t module_t::context() const { return context::detail_::from_handle(context_handle_); }
inline device_t module_t::device() const { return device::get(context::detail_::get_device_id(context_handle_)); }

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_MODULE_HPP_

