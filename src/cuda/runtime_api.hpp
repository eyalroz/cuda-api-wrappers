/**
 * @file
 *
 * @brief A single file which includes, in turn, all of the CUDA
 * Runtime API wrappers and related headers.
 *
 * @note This header includes a subset of the overall API wrapper code;
 * but note that, indirectly, additional headers are included including
 * driver-related ones.
 */
#pragma once
#ifndef CUDA_RUNTIME_API_WRAPPERS_HPP_
#define CUDA_RUNTIME_API_WRAPPERS_HPP_

#include "api/types.hpp"
#include "api/array.hpp"
#include "api/constants.hpp"
#include "api/error.hpp"
#include "api/versions.hpp"
#include "api/miscellany.hpp"
#include "api/device_properties.hpp"
#include "api/current_device.hpp"
#include "api/memory.hpp"
#include "api/pointer.hpp"
#include "api/texture_view.hpp"
#include "api/ipc.hpp"

#include "api/stream.hpp"
#include "api/device.hpp"
#include "api/event.hpp"

#include "api/peer_to_peer.hpp"
#include "api/devices.hpp"

#include "api/detail/pci_id.hpp"
#include "api/kernels/apriori_compiled.hpp"
#include "api/kernel.hpp"
#include "api/launch_configuration.hpp"
#include "api/kernel_launch.hpp"

#include "api/multi_wrapper_impls/pointer.hpp"
#include "api/multi_wrapper_impls/array.hpp"
#include "api/multi_wrapper_impls/event.hpp"
#include "api/multi_wrapper_impls/device.hpp"
#include "api/multi_wrapper_impls/event.hpp"
#include "api/multi_wrapper_impls/stream.hpp"
#include "api/multi_wrapper_impls/memory.hpp"
#include "api/multi_wrapper_impls/kernel.hpp"
#include "api/multi_wrapper_impls/kernel_launch.hpp"
#include "api/multi_wrapper_impls/apriori_compiled_kernel.hpp"

#endif // CUDA_RUNTIME_API_WRAPPERS_HPP_
