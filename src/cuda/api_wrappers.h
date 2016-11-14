/**
 * @file Include all headers with CUDA (Runtime) API wrappers.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_H_
#define CUDA_API_WRAPPERS_H_

#include "cuda/api/types.h"
#include "cuda/api/constants.h"
#include "cuda/api/error.hpp"
#include "cuda/api/versions.hpp"
#include "cuda/api/kernel_launch.cuh"
#include "cuda/api/profiling.h"
#include "cuda/api/device_properties.hpp"
#include "cuda/api/device_count.hpp"
#include "cuda/api/device_function.hpp"
#include "cuda/api/memory.hpp"
#include "cuda/api/unique_ptr.hpp"
#include "cuda/api/ipc.hpp"

#include "cuda/api/stream.hpp"
#include "cuda/api/device.hpp"
#include "cuda/api/event.hpp"

#endif /* CUDA_API_WRAPPERS_H_ */
