/**
 * @file api_wrappers.hpp
 *
 * @brief A single file which includes, in turn, all of the CUDA
 * Runtime API wrappers and related headers.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_HPP_
#define CUDA_API_WRAPPERS_HPP_

#include <cuda/api/types.hpp>
#include <cuda/api/constants.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/versions.hpp>
#include <cuda/api/miscellany.hpp>
#include <cuda/api/kernel_launch.cuh>
#include <cuda/api/profiling.hpp>
#include <cuda/api/device_properties.hpp>
#include <cuda/api/device_count.hpp>
#include <cuda/api/current_device.hpp>
#include <cuda/api/device_function.hpp>
#include <cuda/api/memory.hpp>
#include <cuda/api/pointer.hpp>
#include <cuda/api/unique_ptr.hpp>
#include <cuda/api/ipc.hpp>

#include <cuda/api/stream.hpp>
#include <cuda/api/device.hpp>
#include <cuda/api/event.hpp>

#include <cuda/api/peer_to_peer.hpp>

#include <cuda/api/memory_device_impls.hpp>
#include <cuda/api/pci_id_impl.hpp>
#include <cuda/api/multi_wrapper_impls.hpp>

#endif // CUDA_API_WRAPPERS_HPP_ 
