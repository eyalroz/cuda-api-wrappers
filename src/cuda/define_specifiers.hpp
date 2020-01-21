/**
 * @file define_specifiers.hpp
 *
 * @brief Some functions need a specification of their appropriate execution space w.r.t. the
 * CUDA device-vs-host-side side, as well as their inlining requirement. For brevity,
 * we introduce shorthands for these, to be defined when opening the library's namespace.
 */

#ifdef __CUDACC__

#ifndef CUDA_FD
#define CUDA_FD  __forceinline__ __device__
#endif

#ifndef CUDA_FH
#define CUDA_FH  __forceinline__ __host__
#endif

#ifndef CUDA_FHD
#define CUDA_FHD __forceinline__ __host__ __device__
#endif

#ifndef CUDA_HD
#define CUDA_HD __host__ __device__
#endif

#ifndef CUDA_D
#define CUDA_D __device__
#endif

#ifndef CUDA_H
#define CUDA_H __host__
#endif

#else // __CUDACC__

#ifndef CUDA_FD
#define CUDA_FD inline
#endif

#ifndef CUDA_FH
#define CUDA_FH inline
#endif

#ifndef CUDA_FHD
#define CUDA_FHD inline
#endif

#ifndef CUDA_HD
#define CUDA_HD
#endif

#ifndef CUDA_D
#define CUDA_D
#endif

#ifndef CUDA_H
#define CUDA_H
#endif

#endif // __CUDACC__
