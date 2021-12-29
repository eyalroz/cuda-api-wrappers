/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Device code
extern "C" __global__ void memMapIpc_kernel(char *ptr, int sz, char val)
{
    // Dummy kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < sz; idx += (gridDim.x * blockDim.x)) {
        ptr[idx] = val;
    }
}
