/**
 * @file
 *
 * @brief A micro-benchmark for several implementations of an `isfinite()`
 * function for half-precision floating-point (FP16) values.
 *
 * @note See https://stackoverflow.com/q/79937542/1593077
 */

#include <cuda/api.hpp>
#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>

enum FinityCheckMethod {
    smaller_than_infinity_literal,
    using_cuda_intrinsic_for_float_type,
    less_than_ushort_threshold_value_for_non_finite,
    bit_pattern_comparison,
    bit_pattern_comparison_via_destructuring,
    self_subtraction,
};

#define __fd__ __forceinline__ __device__

template <FinityCheckMethod FCM>
__fd__ bool isfinite (half x);

template <> __fd__ bool isfinite<smaller_than_infinity_literal>(half x)
{
    return __habs(x) < ((half)INFINITY);
}

template <> __fd__ bool isfinite<using_cuda_intrinsic_for_float_type>(half x)
{
    return isfinite(__half2float(x));
}

template <> __fd__ bool isfinite<less_than_ushort_threshold_value_for_non_finite>(half x)
{
    unsigned short shifted_plus_infinity_rep = 0b0111110000000000u << 1;
    unsigned short shifted_input_rep = __half_as_ushort (x) << 1;
    return shifted_input_rep < shifted_plus_infinity_rep;
}

template <> __fd__ bool isfinite<bit_pattern_comparison>(half x)
{
    auto exponent_bits_mask = 0b0111110000000000u; // 10 least-significant bits are the mantissa
    return (__half_as_ushort(x) & exponent_bits_mask) != exponent_bits_mask;
}

template <> __fd__ bool isfinite<bit_pattern_comparison_via_destructuring>(half x)
{
    union destructured_half {
        half value;
        struct {
            unsigned short mantissa: 10;
            unsigned short exponent : 5;
            unsigned short sign : 1;
        } components;
    };
    destructured_half d;
    d.value = x;
    return d.components.exponent != 0b11111u; // all ones
}

template <> __fd__ bool isfinite<self_subtraction>(half x)
{
    return x - x == (half) 0;
}

using count_t = unsigned int;

template <FinityCheckMethod FCM>
__global__ void kernel(count_t *finity_counts)
{
    auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned short arg = tid;
    count_t res = 0;
    // one iteration for every one of the 2^16 possible ushort values
    do {
        res = res + isfinite<FCM>(__ushort_as_half(arg));
        arg++;
    } while (arg != tid);
    finity_counts[tid] = res;
}

const char* name_of(FinityCheckMethod FCM)
{
    static constexpr const char* names[] = {
        "smaller then infinity literal",
        "using cuda intrinsic for float type",
        "less than ushort threshold value for non-finite",
        "direct exponent bits comparison",
        "exponent bits comparison via destructuring",
        "self-substraction criterion",
    };
    static constexpr auto num_names = sizeof(names)/sizeof(char const*);
    return (FCM >= num_names) ? nullptr : names[FCM];
}

template <FinityCheckMethod FCM>
void sanity_check(cuda::span<count_t> d_finity_counts, cuda::span<count_t> finity_counts)
{
    for (size_t tid = 0; tid < d_finity_counts.size(); tid++) {
        auto num_encodings_of_finite_half_values { 0xf800u };
        if (finity_counts[tid] != num_encodings_of_finite_half_values) {
            std::cout
                << "sanity check for check variant " << name_of(FCM) << " failed: "
                << "Thread " << tid << "'s counted " << finity_counts[tid]
                << " finite representations rather than " <<  num_encodings_of_finite_half_values;
            exit(EXIT_FAILURE);
        }
    }
}

template <FinityCheckMethod FCM>
void time_method(cuda::device_t const& device)
{
    auto threads_per_block { 128 };
    auto num_ushort_values { 1lu << sizeof(unsigned short) * CHAR_BIT };
    assert(num_ushort_values % threads_per_block == 0);
    auto num_iterations { 10 };

    auto launch_config = cuda::launch_config_builder()
        .overall_size(num_ushort_values)
        .block_size(threads_per_block)
        .build();

    auto total_threads = num_ushort_values;

    auto d_finity_counts = cuda::make_unique_span<count_t>(device, total_threads);
    auto finity_counts = cuda::make_unique_span<count_t>(total_threads);

    cuda::memory::zero(d_finity_counts);
    auto events = std::make_pair(cuda::event::create(device), cuda::event::create(device));
    auto timings = cuda::generate_unique_span<cuda::event::duration_t>(num_iterations,
        [&](size_t) {
            events.first.record();
            cuda::launch(kernel<FCM>, device, launch_config, d_finity_counts.data());
            events.second.record();
            device.synchronize();
            return cuda::event::time_elapsed_between(events.first, events.second);
        });
    std::nth_element(timings.begin(), timings.begin() + num_iterations / 2, timings.end());
    std::cout << "isfinite check method  " << std::left << std::setw(50) << name_of(FCM)
        << " : " << std::setw(12) << timings[num_iterations / 2].count() << " msec\n";
    cuda::memory::copy(finity_counts, d_finity_counts);

    sanity_check<FCM>(d_finity_counts, finity_counts);
}

int main()
{
    auto device = cuda::device::current::get();
    time_method<smaller_than_infinity_literal>(device);
    time_method<using_cuda_intrinsic_for_float_type>(device);
    time_method<less_than_ushort_threshold_value_for_non_finite>(device);
    time_method<bit_pattern_comparison>(device);
    time_method<bit_pattern_comparison_via_destructuring>(device);
    time_method<self_subtraction>(device);
}
