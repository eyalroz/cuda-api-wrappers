/**
 * @file undefine_specifiers.hpp
 *
 * @brief A "preprocessor utility" header for undefining the CUDA function declaration specifier shorthands,
 * defined in @ref define_specifiers.hpp .
 */
#ifdef __CUDACC__

#ifndef __fd__
#define __fd__  __forceinline__ __device__
#endif

#ifndef __fh__
#define __fh__  __forceinline__ __host__
#endif

#ifndef __fhd__
#define __fhd__ __forceinline__ __host__ __device__
#endif

#ifndef __hd__
#define __hd__ __host__ __device__
#endif

#else // __CUDACC__

#ifndef __fd__
#define __fd__ inline
#endif

#ifndef __fh__
#define __fh__ inline
#endif

#ifndef __fhd__
#define __fhd__ inline
#endif

#ifndef __hd__
#define __hd__
#endif

#endif // __CUDACC__
