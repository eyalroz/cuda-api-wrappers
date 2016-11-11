#pragma once
#ifndef SRC_CUDA_ON_DEVICE_PTX_CUH_
#define SRC_CUDA_ON_DEVICE_PTX_CUH_

#ifndef STRINGIFY
#define STRINGIFY(_q) #_q
#endif

namespace ptx {
namespace special_registers {


#define DEFINE_SPECIAL_REGISTER_GETTER(special_register_name, value_type, width_in_bits) \
__forceinline__ __device__ value_type special_register_name () \
{ \
	value_type ret;  \
	if (std::is_unsigned<value_type>::value) { \
		asm volatile ("mov.u" STRINGIFY(width_in_bits) " %0, %" STRINGIFY(special_register_name) ";" : "=r"(ret)); \
	} \
	else { \
		asm volatile ("mov.s" STRINGIFY(width_in_bits) " %0, %" STRINGIFY(special_register_name) ";" : "=r"(ret)); \
	} \
	return ret; \
} \

DEFINE_SPECIAL_REGISTER_GETTER( laneid,             unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( warpid,             unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( gridid,             unsigned long long, 64);
DEFINE_SPECIAL_REGISTER_GETTER( smid,               unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( nsmid,              unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( clock,              unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( clock64,            unsigned long long, 64);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_lt,        unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_le,        unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_eq,        unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_ge,        unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_gt,        unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( dynamic_smem_size,  unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( total_smem_size,    unsigned,           32);

#undef DEFINE_SPECIAL_REGISTER_GETTER

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
