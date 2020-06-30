/**
 * @file api_wrappers.hpp
 *
 * @brief A single file which includes, in turn, all of the CUDA
 * Runtime API wrappers and related headers.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_HPP_
#define CUDA_API_WRAPPERS_HPP_

static_assert(__cplusplus >= 201103L, "The CUDA Runtime API headers can only be compiled with C++11 or a later version of the C++ language standard");

#include <cuda/api/array.hpp>
#include <cuda/api/types.hpp>
#include <cuda/api/constants.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/versions.hpp>
#include <cuda/api/miscellany.hpp>
#include <cuda/api/profiling.hpp>
#include <cuda/api/device_properties.hpp>
#include <cuda/api/current_device.hpp>
#include <cuda/api/memory.hpp>
#include <cuda/api/pointer.hpp>
#include <cuda/api/texture_view.hpp>
#include <cuda/api/unique_ptr.hpp>
#include <cuda/api/ipc.hpp>

#include <cuda/api/stream.hpp>
#include <cuda/api/device.hpp>
#include <cuda/api/event.hpp>

#include <cuda/api/peer_to_peer.hpp>
#include <cuda/api/devices.hpp>

#include <cuda/api/pci_id_impl.hpp>
#include <cuda/api/multi_wrapper_impls.hpp>
#include "api/kernel.hpp"
#include "api/kernel_launch.hpp"

#endif // CUDA_API_WRAPPERS_HPP_
