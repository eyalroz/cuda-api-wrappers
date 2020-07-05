#pragma once
#ifndef SRC_CUDA_ON_DEVICE_PTX_CUH_
#define SRC_CUDA_ON_DEVICE_PTX_CUH_

#include <cstdint>

#ifndef STRINGIFY
#define STRINGIFY(_q) #_q
#endif

namespace ptx {
namespace special_registers {


#define SIZE_MARKER_u16 "h"
#define SIZE_MARKER_u32 "r"
#define SIZE_MARKER_u64 "l"
#define SIZE_MARKER_f32 "f"
#define SIZE_MARKER_f64 "d"

#define SIZE_MARKER(kind_of_register) SIZE_MARKER_ ## kind_of_register

#define RETURN_TYPE_u16 std::uint16_t
#define RETURN_TYPE_u32 std::uint32_t
#define RETURN_TYPE_u64 std::uint64_t
#define RETURN_TYPE_f32 float
#define RETURN_TYPE_f64 double

#define RETURN_TYPE(kind_of_register) RETURN_TYPE_ ## kind_of_register

#define DEFINE_SPECIAL_REGISTER_GETTER(special_register_name, kind_of_register) \
__forceinline__ __device__ RETURN_TYPE(kind_of_register) special_register_name() \
{ \
	RETURN_TYPE(kind_of_register) ret;  \
	asm volatile ("mov." STRINGIFY(kind_of_register) "%0, %" STRINGIFY(special_register_name) ";" : "=" SIZE_MARKER(kind_of_register) (ret)); \
	return ret; \
} \

DEFINE_SPECIAL_REGISTER_GETTER( laneid,             u32);
DEFINE_SPECIAL_REGISTER_GETTER( warpid,             u32);
DEFINE_SPECIAL_REGISTER_GETTER( gridid,             u64);
DEFINE_SPECIAL_REGISTER_GETTER( smid,               u32);
DEFINE_SPECIAL_REGISTER_GETTER( nsmid,              u32);
DEFINE_SPECIAL_REGISTER_GETTER( clock,              u32);
DEFINE_SPECIAL_REGISTER_GETTER( clock64,            u64);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_lt,        u32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_le,        u32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_eq,        u32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_ge,        u32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_gt,        u32);
DEFINE_SPECIAL_REGISTER_GETTER( dynamic_smem_size,  u32);
DEFINE_SPECIAL_REGISTER_GETTER( total_smem_size,    u32);

#undef DEFINE_SPECIAL_REGISTER_GETTER
#undef RETURN_TYPE
#undef RETURN_TYPE_u16
#undef RETURN_TYPE_u32
#undef RETURN_TYPE_u64
#undef RETURN_TYPE_f32
#undef RETURN_TYPE_f64
#undef SIZE_MARKER
#undef SIZE_MARKER_u16
#undef SIZE_MARKER_u32
#undef SIZE_MARKER_u64
#undef SIZE_MARKER_f32
#undef SIZE_MARKER_f64


/*
 * Not defining getters for:
 *
 * %tid                      - available as threadIdx
 * %ntid                     - available as blockDim
 * %warpid                   - not interesting
 * %nwarpid                  - not interesting
 * %ctaid                    - available is blockId
 * %nctaid                   - available as gridDim
 * %pm0, ..., %pm7           - not interesting, for now (performance monitoring)
 * %pm0_64, ..., %pm7_64     - not interesting, for now (performance monitoring)
 * %envreg0, ..., %envreg31  - not interesting, for now (performance monitoring)
 */


} // namespace special_registers


template <typename T>
__forceinline__ __device__ T ldg(const T* ptr)
{
#if __CUDA_ARCH__ >= 320
	return __ldg(ptr);
#else
	return *ptr; // maybe we should ld.cg or ld.cs here?
#endif
}
template <typename T>
__forceinline__ __device__ T load_global_with_non_coherent_cache(const T* ptr) { return ldg(ptr); }

} // namespace ptx

#endif /* SRC_CUDA_ON_DEVICE_PTX_CUH_ */
