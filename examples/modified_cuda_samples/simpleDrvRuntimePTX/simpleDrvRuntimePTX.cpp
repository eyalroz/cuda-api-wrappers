/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It loads a cuda fatbinary and runs vector addition kernel.
 * Uses both Driver and Runtime CUDA APIs for different purposes.
 */

#include "../../common.hpp"
#include <random>

std::string create_ptx_file()
{
	const char* ptx_file_contents = R"(
	.version 6.5
	.target sm_30
	.address_size 64

		// .globl   dummy
	
	.visible .entry dummy(
	
)
	{
		ret;
	}
	

		// .globl	VecAdd_kernel

	.visible .entry VecAdd_kernel(
		.param .u64 VecAdd_kernel_param_0,
		.param .u64 VecAdd_kernel_param_1,
		.param .u64 VecAdd_kernel_param_2,
		.param .u32 VecAdd_kernel_param_3
	)
	{
		.reg .pred 	%p<2>;
		.reg .f32 	%f<4>;
		.reg .b32 	%r<6>;
		.reg .b64 	%rd<11>;


		ld.param.u64 	%rd1, [VecAdd_kernel_param_0];
		ld.param.u64 	%rd2, [VecAdd_kernel_param_1];
		ld.param.u64 	%rd3, [VecAdd_kernel_param_2];
		ld.param.u32 	%r2, [VecAdd_kernel_param_3];
		mov.u32 	%r3, %ntid.x;
		mov.u32 	%r4, %ctaid.x;
		mov.u32 	%r5, %tid.x;
		mad.lo.s32 	%r1, %r4, %r3, %r5;
		setp.ge.s32	%p1, %r1, %r2;
		@%p1 bra 	BB0_2;

		cvta.to.global.u64 	%rd4, %rd1;
		mul.wide.s32 	%rd5, %r1, 4;
		add.s64 	%rd6, %rd4, %rd5;
		cvta.to.global.u64 	%rd7, %rd2;
		add.s64 	%rd8, %rd7, %rd5;
		ld.global.f32 	%f1, [%rd8];
		ld.global.f32 	%f2, [%rd6];
		add.f32 	%f3, %f2, %f1;
		cvta.to.global.u64 	%rd9, %rd3;
		add.s64 	%rd10, %rd9, %rd5;
		st.global.f32 	[%rd10], %f3;

	BB0_2:
		ret;
	}
	)";

#if _POSIX_C_SOURCE >= 200809L
	char temp_filename[] = "caw-simple-drv-runtime-ptx-XXXXXX";
	int file_descriptor = mkstemp(temp_filename);
	if (file_descriptor == -1) {
		throw std::runtime_error(std::string("Failed creating a temporary file using mkstemp(): ") + std::strerror(errno) + '\n');
	}
	FILE* ptx_file = fdopen(file_descriptor, "w");
#else
	char temp_filename[L_tmpnam];
	std::tmpnam(temp_filename);
	FILE* ptx_file = fopen(temp_filename, "w");
#endif
	if (ptx_file == nullptr) {
        throw std::runtime_error(std::string("Failed converting temporay file descriptor into a C library FILE structure: ") + std::strerror(errno) + '\n');
    }
	if (fputs(ptx_file_contents, ptx_file) == EOF) {
        throw std::runtime_error("Failed writing PTX to temporary file " + std::string(temp_filename) + ": " + std::strerror(errno) + '\n');
    }
	if (fclose(ptx_file) == EOF) {
        throw std::runtime_error("Failed closing temporary PTX file " + std::string(temp_filename) + ": " + std::strerror(errno) + '\n');
    }
	return temp_filename;
}

// Host code
int main(int argc, char** argv)
{
    std::cout << "simpleDrvRuntime - PTX version.\n";
    int N = 50000;
    size_t  size = N * sizeof(float);

    auto device_id = choose_device(argc, argv);
    auto device = cuda::device::get(device_id);

    // Create context
    auto context = cuda::context::create(device);

    cuda::context::current::scoped_override_t context_setter { context };

// first search for the module path before we load the results
    auto ptx_filename = create_ptx_file();

    auto module = cuda::module::load_from_file(context, ptx_filename);
    auto vecAdd_kernel = module.get_kernel("VecAdd_kernel");
    auto dummy_kernel = module.get_kernel("dummy");

    auto stream = cuda::stream::create(context, cuda::stream::async);

    stream.enqueue.kernel_launch(dummy_kernel, cuda::launch_configuration_t{1,1});

    cuda::outstanding_error::ensure_none();

    stream.synchronize();

	auto h_A = std::unique_ptr<float[]>(new float[N]);
	auto h_B = std::unique_ptr<float[]>(new float[N]);
	auto h_C = std::unique_ptr<float[]>(new float[N]);

	auto generator = []() {
		static std::random_device random_device;
		static std::mt19937 randomness_generator { random_device() };
		static std::uniform_real_distribution<> distribution { 0.0, 1.0 };
		return distribution(randomness_generator);
	};
	std::generate_n(h_A.get(), N, generator);
	std::generate_n(h_B.get(), N, generator);

    // Allocate vectors in device memory
	auto d_A = cuda::memory::make_unique_span<float>(device, N);
	auto d_B = cuda::memory::make_unique_span<float>(device, N);
	auto d_C = cuda::memory::make_unique_span<float>(device, N);


	cuda::memory::async::copy(d_A, h_A.get(), size, stream);
	cuda::memory::async::copy(d_B, h_B.get(), size, stream);

	auto launch_config = cuda::launch_config_builder()
		.overall_size(N)
		.block_size(256)
		.build();

    cuda::outstanding_error::ensure_none();

    stream.enqueue.kernel_launch(vecAdd_kernel, launch_config, d_A.data(), d_B.data(), d_C.data(), N);

	cuda::memory::async::copy(h_C.get(), d_C, size, stream);
	stream.synchronize();

	for (int i = 0; i < N; ++i) {
		if (std::fabs(h_A.get()[i] + h_B.get()[i] - h_C.get()[i]) > 1e-5f)  {
			std::cerr << "Result verification failed at element " << i << "\n";
			exit(EXIT_FAILURE);
		}
	}
    std::cout << "SUCCESS\n";
    return EXIT_SUCCESS;
}
