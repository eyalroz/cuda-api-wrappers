/**
 * Derived from the nVIDIA CUDA 10.0 samples by
 *
 *   Eyal Rozenberg <eyalroz1@gmx.com>
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 *
 * This version differs from the other vectorAdd example in that managed memory is
 * used instead of regular host and device memory.
 */

#include <cuda/rtc.hpp>
#include <cuda/api.hpp>

#include <iostream>
#include <cmath>
#include <random>

const char* vectorAdd_source = R"(

.version 7.1
.target sm_52
.address_size 64

//
// Computes the vector addition of param_0 and param_1 into param_2.
// The three vectors have the same number of elements, param_3
//
.visible .entry _Z9vectorAddPKfS0_Pfi(
        .param .u64 _Z9vectorAddPKfS0_Pfi_param_0,
        .param .u64 _Z9vectorAddPKfS0_Pfi_param_1,
        .param .u64 _Z9vectorAddPKfS0_Pfi_param_2,
        .param .u32 _Z9vectorAddPKfS0_Pfi_param_3
)
{
        .reg .pred      %p<2>;
        .reg .f32       %f<4>;
        .reg .b32       %r<6>;
        .reg .b64       %rd<11>;

        ld.param.u64    %rd1, [_Z9vectorAddPKfS0_Pfi_param_0];
        ld.param.u64    %rd2, [_Z9vectorAddPKfS0_Pfi_param_1];
        ld.param.u64    %rd3, [_Z9vectorAddPKfS0_Pfi_param_2];
        ld.param.u32    %r2, [_Z9vectorAddPKfS0_Pfi_param_3];
        mov.u32         %r3, %ntid.x;
        mov.u32         %r4, %ctaid.x;
        mov.u32         %r5, %tid.x;
        mad.lo.s32      %r1, %r4, %r3, %r5;
        setp.ge.s32     %p1, %r1, %r2;
        @%p1 bra        BB0_2;

        cvta.to.global.u64      %rd4, %rd1;
        mul.wide.s32    %rd5, %r1, 4;
        add.s64         %rd6, %rd4, %rd5;
        cvta.to.global.u64      %rd7, %rd2;
        add.s64         %rd8, %rd7, %rd5;
        ld.global.f32   %f1, [%rd8];
        ld.global.f32   %f2, [%rd6];
        add.f32         %f3, %f2, %f1;
        cvta.to.global.u64      %rd9, %rd3;
        add.s64         %rd10, %rd9, %rd5;
        st.global.f32   [%rd10], %f3;

BB0_2:
        ret;
}

)";

int main(void)
{
	int numElements = 50000;
	size_t size = numElements * sizeof(float);
	auto kernel_name = "vectorAdd";

	std::cout << "[Vector addition of " << numElements << " elements]\n";

	auto device = cuda::device::current::get();
	auto compilation_output = cuda::rtc::program_t<cuda::ptx>(kernel_name)
		.set_source(vectorAdd_source)
		.set_target(device).compile();

	if (compilation_output.failed()) {
		auto log = compilation_output.log();
		std::cout << "Compilation log:\n" << log.data() << "\n---------\n";
	}

	auto context = cuda::device::current::get().primary_context();
	auto module = cuda::module::create(context, compilation_output);
	constexpr const auto mangled_kernel_name = "_Z9vectorAddPKfS0_Pfi";
	auto vectorAdd = module.get_kernel(mangled_kernel_name);

	auto h_A = std::vector<float>(numElements);
	auto h_B = std::vector<float>(numElements);
	auto h_C = std::vector<float>(numElements);

	auto generator = []() {
		static std::random_device random_device;
		static std::mt19937 randomness_generator { random_device() };
		static std::uniform_real_distribution<> distribution { 0.0, 1.0 };
		return distribution(randomness_generator);
	};
	std::generate(h_A.begin(), h_A.end(), generator);
	std::generate(h_B.begin(), h_B.end(), generator);

	auto d_A = cuda::memory::make_unique_span<float>(device, numElements);
	auto d_B = cuda::memory::make_unique_span<float>(device, numElements);
	auto d_C = cuda::memory::make_unique_span<float>(device, numElements);

	cuda::memory::copy(d_A, h_A.data());
	cuda::memory::copy(d_B, h_B.data());

	auto launch_config = cuda::launch_config_builder()
		.overall_size(numElements)
		.block_size(256)
		.build();

	std::cout
		<< "CUDA kernel launch with " << launch_config.dimensions.grid.x
		<< " blocks of " << launch_config.dimensions.block.x << " threads each\n";

	cuda::launch(
		vectorAdd, launch_config,
		d_A.get(), d_B.get(), d_C.get(), numElements
	);

	cuda::memory::copy(h_C.data(), d_C, size);

	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i) {
		if (std::fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5f)  {
			std::cerr << "Result verification failed at element " << i << "\n";
			exit(EXIT_FAILURE);
		}
	}

    std::cout << "Test PASSED\n";
    std::cout << "SUCCESS\n";
}


