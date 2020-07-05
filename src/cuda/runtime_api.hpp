/**
 * @file runtime_api.hpp
 *
 * @brief A single file which includes, in turn, all of the CUDA
 * Runtime API wrappers and related headers.
 */
#pragma once
#ifndef CUDA_RUNTIME_API_WRAPPERS_HPP_
#define CUDA_RUNTIME_API_WRAPPERS_HPP_

static_assert(__cplusplus >= 201103L, "The CUDA Runtime API headers can only be compiled with C++11 or a later version of the C++ language standard");

#include <cuda/common/types.hpp>
#include <cuda/runtime_api/array.hpp>
#include <cuda/runtime_api/constants.hpp>
#include <cuda/runtime_api/error.hpp>
#include <cuda/runtime_api/versions.hpp>
#include <cuda/runtime_api/miscellany.hpp>
#include <cuda/runtime_api/device_properties.hpp>
#include <cuda/runtime_api/current_device.hpp>
#include <cuda/runtime_api/memory.hpp>
#include <cuda/runtime_api/pointer.hpp>
#include <cuda/runtime_api/texture_view.hpp>
#include <cuda/runtime_api/unique_ptr.hpp>
#include <cuda/runtime_api/ipc.hpp>

#include <cuda/runtime_api/stream.hpp>
#include <cuda/runtime_api/device.hpp>
#include <cuda/runtime_api/event.hpp>

#include <cuda/runtime_api/peer_to_peer.hpp>
#include <cuda/runtime_api/devices.hpp>

#include <cuda/runtime_api/pci_id_impl.hpp>
#include <cuda/runtime_api/multi_wrapper_impls.hpp>
#include <cuda/runtime_api/kernel.hpp>
#include <cuda/runtime_api/kernel_launch.hpp>

#endif // CUDA_RUNTIME_API_WRAPPERS_HPP_
