#include <iostream>
#include <numeric>

#include <cuda/api_wrappers.hpp>


namespace kernels {

__global__ void from_3D_texture_to_memory_space(cudaTextureObject_t texture_source, float* destination, size_t w, size_t h, size_t d) {

	const auto gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	const auto gtidy = threadIdx.y + blockIdx.y * blockDim.y;
	const auto gtidz = threadIdx.z + blockIdx.z * blockDim.z;
	const auto gtid = gtidx + gtidy * w + gtidz * w * h;

	if (gtidx < w && gtidy < h && gtidz < d) {
		destination[gtid] = tex3D<float>(texture_source, gtidx, gtidy, gtidz);
	}
}

__global__ void from_2D_texture_to_memory_space(cudaTextureObject_t texture_source, float* destination, size_t w, size_t h) {

	const auto gtidx = threadIdx.x + blockIdx.x * blockDim.x;
	const auto gtidy = threadIdx.y + blockIdx.y * blockDim.y;
	const auto gtid = gtidx + gtidy * static_cast<unsigned>(w);

	if (gtidx < w && gtidy < h) {
		const float x = tex2D<float>(texture_source, gtidx, gtidy);
		printf("thread %u %u, reading value %f, and writing to index %u\n", gtidx, gtidy, x, gtid);
		destination[gtid] = x;
	}
}

}

size_t div_ceil(size_t dividend, size_t divisor) {
	return dividend / divisor + (dividend % divisor == 0 ? 0 : 1);
}

template<class Device>
void array_3d_example(Device& device, size_t w, size_t h, size_t d) {
	const cuda::array::dimensions_t<3> dims = {w, h, d};

	cuda::array::array_t<float, 3> arr(device, dims);
	auto ptr_in = cuda::memory::managed::make_unique<float[]>(arr.size());
	std::iota(ptr_in.get(), ptr_in.get() + arr.size(), 0);
	auto ptr_out = cuda::memory::managed::make_unique<float[]>(arr.size());
	cuda::memory::copy(arr, ptr_in.get());
	cuda::texture_view tv(arr);
	constexpr cuda::grid_block_dimension_t block_dim = 10;
	constexpr cuda::grid_block_dimensions_t block_dims = {block_dim, block_dim, block_dim};
	const cuda::grid_dimensions_t grid_dims = {div_ceil(w, block_dim), div_ceil(h, block_dim), div_ceil(d, block_dim)};
	cuda::launch(kernels::from_3D_texture_to_memory_space, cuda::make_launch_config(grid_dims, block_dims), tv.get(), ptr_out.get(), w, h, d);
	device.synchronize();
	for (size_t i = 0; i < arr.size(); ++i) {
		if (ptr_out[i] != i) {
			std::cout << "ERROR: values were not correctly copied!" << std::endl;
		}
	}

	// copy between arrays and memory spaces
	cuda::array::array_t<float, 3> other_arr(device, dims);
	cuda::memory::copy(other_arr, ptr_out.get());
	cuda::memory::copy(ptr_in.get(), other_arr);

	for (size_t i = 0; i < arr.size(); ++i) {
		if (ptr_in[i] != i) {
			std::cout << "ERROR: values were not correctly copied!" << std::endl;
		}
	}

	// also asynchronously
	auto stream = device.create_stream(cuda::stream::async);
	cuda::memory::async::copy(other_arr, ptr_out.get(), stream);
	cuda::memory::async::copy(ptr_in.get(), other_arr, stream);
	device.synchronize();
	for (size_t i = 0; i < arr.size(); ++i) {
		if (ptr_in[i] != i) {
			std::cout << "ERROR: values were not correctly copied!" << std::endl;
		}
	}
}

template<class Device>
void array_2d_example(Device& device, size_t w, size_t h) {

	const cuda::array::dimensions_t<2> dims = {w, h};
	cuda::array::array_t<float, 2> arr(device , dims);
	auto ptr_in = cuda::memory::managed::make_unique<float[]>(arr.size());
	std::iota(ptr_in.get(), ptr_in.get() + arr.size(), 0);
	auto ptr_out = cuda::memory::managed::make_unique<float[]>(arr.size());

	for (size_t i = 0; i < h; ++i) {
		for (size_t j = 0; j < w; ++j) {
			std::cout << ptr_in[j + i * w] << ' ';
		}
		std::cout << std::endl;
	}

	cuda::memory::copy(arr, ptr_in.get());
	cuda::texture_view tv(arr);

	constexpr cuda::grid_block_dimension_t block_dim = 10;
	constexpr cuda::grid_block_dimensions_t block_dims = {block_dim, block_dim, 1};
	const cuda::grid_dimensions_t grid_dims = {div_ceil(w, block_dim), div_ceil(h, block_dim), 1};

	cuda::launch(kernels::from_2D_texture_to_memory_space, cuda::make_launch_config(grid_dims, block_dims), tv.get(), ptr_out.get(), w, h);
	device.synchronize();
	for (size_t i = 0; i < h; ++i) {
		for (size_t j = 0; j < w; ++j) {
			std::cout << ptr_out[j + i * w] << ' ';
		}
		std::cout << std::endl;
	}
	for (size_t k = 0; k < arr.size(); ++k) {
		if (ptr_out[k] != k) {
			std::cout << "ERROR: values were not correctly copied!" << std::endl;
		}
	}

	// copy between arrays and memory spaces
	cuda::array::array_t<float, 2> other_arr(device, dims);
	cuda::memory::copy(other_arr, ptr_out.get());
	cuda::memory::copy(ptr_in.get(), other_arr);
	for (size_t k = 0; k < arr.size(); ++k) {
		if (ptr_in[k] != k) {
			std::cout << "ERROR: values were not correctly copied!" << std::endl;
		}
	}
	
	// also asynchronously
	auto stream = device.create_stream(cuda::stream::async);
	cuda::memory::async::copy(other_arr, ptr_out.get(), stream);
	cuda::memory::async::copy(ptr_in.get(), other_arr, stream);
	device.synchronize();
	for (size_t k = 0; k < arr.size(); ++k) {
		if (ptr_in[k] != k) {
			std::cout << "ERROR: values were not correctly copied!" << std::endl;
		}
	}
}

int main() {

	auto device = cuda::device::current::get();

	// array dimensions
	constexpr size_t w = 3;
	constexpr size_t h = 3;
	constexpr size_t d = 3;

	array_3d_example(device, w, h, d);
	array_2d_example(device, w, h);
}
