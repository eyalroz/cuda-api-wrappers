#include <cuda/api_wrappers.hpp>

#include <iostream>

__global__ void from_texture_to_ptr_kernel3D(cudaTextureObject_t texture_obj, float* ptr, size_t w, size_t h, size_t d) {

	const auto gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	const auto gtidy = threadIdx.y + blockIdx.y * blockDim.y;
	const auto gtidz = threadIdx.z + blockIdx.z * blockDim.z;
	const auto gtid = gtidx + gtidy * w + gtidz * w * h;

	if (gtidx < w && gtidy < h && gtidz < d) {
		ptr[gtid] = tex3D<float>(texture_obj, gtidx, gtidy, gtidz);
	}
}

__global__ void from_texture_to_ptr_kernel2D(cudaTextureObject_t texture_obj, float* ptr, size_t w, size_t h) {

	const auto gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	const auto gtidy = threadIdx.y + blockIdx.y * blockDim.y;
	const auto gtid = gtidx + gtidy * w;

	if (gtidx < w && gtidy < h) {
		auto x = tex2D<float>(texture_obj, gtidx, gtidy);
		printf("thread %u %u, reading value %f, and writing to index %u\n", gtidx, gtidy, x, gtid);
		ptr[gtid] = x;
	}
}


int main() {

	auto device = cuda::device::current::get();

	//
	// 3D
	//
	constexpr size_t w = 3;
	constexpr size_t h = 3;
	constexpr size_t d = 3;
	cuda::array::dimensions_t<3> dims = {w, h, d};

	cuda::array::array_t<float, 3> arr0(device, dims);
	auto ptr_in0 = cuda::memory::managed::make_unique<float[]>(arr0.size());
	ptr_in0[5] = 6;
	auto ptr_out0 = cuda::memory::managed::make_unique<float[]>(arr0.size());
	cuda::memory::copy(arr0, ptr_in0.get());
	cuda::texture::texture_view tv0(arr0);
	cuda::launch(from_texture_to_ptr_kernel3D, cuda::make_launch_config(1, {10, 10, 10}), tv0.get(), ptr_out0.get(), w, h, d);
	device.synchronize();
	for (size_t i = 0; i < arr0.size(); ++i) {
		std::cout << "ptr_out0[" << i << "] = " << ptr_out0[i] << std::endl;
	}

	//
	// 2D
	//
	cuda::array::array_t<float, 2> arr1(device , {w, h});
	auto ptr_in1 = cuda::memory::managed::make_unique<float[]>(arr1.size());
	ptr_in1[0] = 1;
	ptr_in1[1] = 2;
	ptr_in1[3] = 3;
	auto ptr_out1 = cuda::memory::managed::make_unique<float[]>(arr1.size());

	std::cout << std::endl;
	for (size_t i = 0; i < h; ++i) {
		for (size_t j = 0; j < w; ++j) {
			std::cout << ptr_in1[j + i * w] << ' ';
		}
		std::cout << std::endl;
	}

	cuda::memory::copy(arr1, ptr_in1.get());
	cuda::texture::texture_view tv1(arr1);

	cuda::launch(from_texture_to_ptr_kernel2D, cuda::make_launch_config(1, {10, 10, 1}), tv1.get(), ptr_out1.get(), w, h);
	cuda::memory::copy(ptr_out1.get(), arr1);
	device.synchronize();
	for (size_t i = 0; i < h; ++i) {
		for (size_t j = 0; j < w; ++j) {
			std::cout << ptr_out1[j + i * w] << ' ';
		}
		std::cout << std::endl;
	}

	//
	// copy between arrays and pointers
	//
	cuda::array::array_t<float, 3> arr2(device, dims);
	cuda::memory::copy(arr2, ptr_out0.get());
	cuda::memory::copy(ptr_in0.get(), arr2);

	// also asynchronously
	auto stream = device.create_stream(cuda::stream::async);
	cuda::memory::async::copy(arr2, ptr_out0.get(), stream);
	cuda::memory::async::copy(ptr_in0.get(), arr2, stream);

}
