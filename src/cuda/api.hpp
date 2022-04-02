/**
 * @file
 *
 * @brief A single file which includes, in turn, all (joint)
 * wrappers for Runtime and Driver APIs, and related headers.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_HPP_
#define CUDA_API_WRAPPERS_HPP_

static_assert(__cplusplus >= 201103L, "The CUDA API headers can only be compiled with C++11 or a later version of the C++ language standard");

#include <cuda/api/types.hpp>

#include <cuda/api/pci_id.hpp>
#include <cuda/api/constants.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/versions.hpp>
#include <cuda/api/miscellany.hpp>
#include <cuda/api/pointer.hpp>
#include <cuda/api/device_properties.hpp>
#include <cuda/api/current_context.hpp>
#include <cuda/api/ipc.hpp>
#include <cuda/api/array.hpp>
#include <cuda/api/texture_view.hpp>
#include <cuda/api/memory.hpp>
#include <cuda/api/unique_ptr.hpp>
#include <cuda/api/link_options.hpp>

#include <cuda/api/device.hpp>
#include <cuda/api/context.hpp>
#include <cuda/api/primary_context.hpp>
#include <cuda/api/stream.hpp>
#include <cuda/api/event.hpp>
#include <cuda/api/kernel.hpp>
#include <cuda/api/module.hpp>
#include <cuda/api/link.hpp>

#include <cuda/api/current_device.hpp>

#include <cuda/api/peer_to_peer.hpp>
#include <cuda/api/devices.hpp>

#include <cuda/api/pci_id_impl.hpp>
#include <cuda/api/apriori_compiled_kernel.hpp>
#include <cuda/api/kernel_launch.hpp>
#include <cuda/api/virtual_memory.hpp>

#include <cuda/api/multi_wrapper_impls/pointer.hpp>
#include <cuda/api/multi_wrapper_impls/array.hpp>
#include <cuda/api/multi_wrapper_impls/event.hpp>
#include <cuda/api/multi_wrapper_impls/device.hpp>
#include <cuda/api/multi_wrapper_impls/event.hpp>
#include <cuda/api/multi_wrapper_impls/context.hpp>
#include <cuda/api/multi_wrapper_impls/stream.hpp>
#include <cuda/api/multi_wrapper_impls/memory.hpp>
#include <cuda/api/multi_wrapper_impls/kernel.hpp>
#include <cuda/api/multi_wrapper_impls/kernel_launch.hpp>
#include <cuda/api/multi_wrapper_impls/apriori_compiled_kernel.hpp>
#include <cuda/api/multi_wrapper_impls/module.hpp>

#endif // CUDA_API_WRAPPERS_HPP_
