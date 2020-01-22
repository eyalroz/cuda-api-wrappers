#include <cuda/api_wrappers.hpp>

#include <iostream>
#include <iomanip>
#include <numeric>
#include <limits>

using std::size_t;

namespace kernels {

__global__ void from_3D_texture_to_memory_space(
	cudaTextureObject_t texture_source, float* destination, size_t w, size_t h, size_t d)
{
	const uint3 gtid = {
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y,
		threadIdx.z + blockIdx.z * blockDim.z
	};

	const auto gtid_serialized = gtid.x + gtid.y * w + gtid.z * w * h;

	if (gtid.x < w && gtid.y < h && gtid.z < d) {
		destination[gtid_serialized] = tex3D<float>(texture_source, gtid.x, gtid.y, gtid.z);
	}
}

__global__ void from_2D_texture_to_memory_space(cudaTextureObject_t texture_source, float* destination, size_t w, size_t h) {

	const uint2 gtid = {
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y
	};
	const auto gtid_serliazed = gtid.x + gtid.y * static_cast<unsigned>(w);

	if (gtid.x < w && gtid.y < h) {
		const float x = tex2D<float>(texture_source, gtid.x, gtid.y);
		printf("Thread %u %u, reading value %4f, and writing to index %3u\n", gtid.x, gtid.y, x, gtid_serliazed);
		destination[gtid_serliazed] = x;
	}
}

} // namespace kernels

template <typename I, typename I2>
constexpr I div_rounding_up(I dividend, const I2 divisor) noexcept
{
	return (dividend / divisor) + !!(dividend % divisor);
}

template <typename T>
void check_output_is_iota(std::string name, const T* actual, size_t length) noexcept
{
	bool failed { false };
	for (size_t i = 0; i < length; ++i) {
		if (actual[i] != i) {
			if (not failed) {
				std::cerr << name << ": Output does not matched expected values:\n";
			}
			std::cerr << "output[" << std::setw(3) << i << "] = " << actual[i] << " != " << i << '\n';
		}
	}
	if (failed) { std::cerr << '\n'; }
}

template<class Device>
void array_3d_example(Device& device, size_t w, size_t h, size_t d) {
	namespace grid = cuda::grid;

	const cuda::array::dimensions_t<3> dims = {w, h, d};
	cuda::array_t<float, 3> arr(device, dims);
	auto ptr_in = cuda::memory::managed::make_unique<float[]>(arr.size());
	std::iota(ptr_in.get(), ptr_in.get() + arr.size(), 0);
	auto ptr_out = cuda::memory::managed::make_unique<float[]>(arr.size());
	cuda::memory::copy(arr, ptr_in.get());
	cuda::texture_view tv(arr);
	constexpr cuda::grid::block_dimension_t block_dim = 10;
	constexpr auto block_dims = cuda::grid::block_dimensions_t::cube(block_dim);
	assert(div_rounding_up(w, block_dim) <= std::numeric_limits<grid::dimension_t>::max());
	assert(div_rounding_up(h, block_dim) <= std::numeric_limits<grid::dimension_t>::max());
	assert(div_rounding_up(d, block_dim) <= std::numeric_limits<grid::dimension_t>::max());
	const grid::dimensions_t grid_dims = {
		grid::dimension_t( div_rounding_up(w, block_dim) ),
		grid::dimension_t( div_rounding_up(h, block_dim) ),
		grid::dimension_t( div_rounding_up(d, block_dim) )
	};
	cuda::launch(
		kernels::from_3D_texture_to_memory_space,
		cuda::make_launch_config(grid_dims, block_dims),
		tv.get(), ptr_out.get(), w, h, d);
	device.synchronize();
	check_output_is_iota("copy from 3D texture into (managed) global memory", ptr_out.get(), arr.size());

	// copy between arrays and memory spaces
	cuda::array_t<float, 3> other_arr(device, dims);
	cuda::memory::copy(other_arr, ptr_out.get());
	cuda::memory::copy(ptr_in.get(), other_arr);

	check_output_is_iota("copy from (managed) global memory into a 3D array", ptr_in.get(), arr.size());

	// also asynchronously
	auto stream = device.create_stream(cuda::stream::async);
	cuda::memory::async::copy(other_arr, ptr_out.get(), stream);
	cuda::memory::async::copy(ptr_in.get(), other_arr, stream);
	device.synchronize();
	check_output_is_iota("copy from (managed) global memory into a 3D array, asynchronously", ptr_in.get(), arr.size());
}

template <typename T>
void print_2d_array(const T* a, size_t width, size_t height)
{
	for (size_t i = 0; i < height; ++i) {
		for (size_t j = 0; j < width; ++j) {
			std::cout << a[j + i * width] << ' ';
		}
		std::cout << '\n';
	}
}

template<class Device>
void array_2d_example(Device& device, size_t w, size_t h)
{
	namespace grid = cuda::grid;

	const cuda::array::dimensions_t<2> dims = {w, h};
	cuda::array_t<float, 2> arr(device , dims);
	auto ptr_in = cuda::memory::managed::make_unique<float[]>(arr.size());
	std::iota(ptr_in.get(), ptr_in.get() + arr.size(), 0);
	auto ptr_out = cuda::memory::managed::make_unique<float[]>(arr.size());

	std::cout << std::endl;

	print_2d_array(ptr_in.get(), w, h);

	cuda::memory::copy(arr, ptr_in.get());
	cuda::texture_view tv(arr);

	constexpr cuda::grid::block_dimension_t block_dim = 10;
	constexpr auto block_dims = cuda::grid::block_dimensions_t::square(block_dim);
	assert(div_rounding_up(w, block_dim) <= std::numeric_limits<grid::dimension_t>::max());
	assert(div_rounding_up(h, block_dim) <= std::numeric_limits<grid::dimension_t>::max());
	const cuda::grid::dimensions_t grid_dims = {
		grid::dimension_t( div_rounding_up(w, block_dim) ),
		grid::dimension_t( div_rounding_up(h, block_dim) ),
		1
	};

	cuda::launch(
		kernels::from_2D_texture_to_memory_space,
		cuda::make_launch_config(grid_dims, block_dims),
		tv.get(), ptr_out.get(), w, h);
	cuda::memory::copy(ptr_out.get(), arr);
	device.synchronize();
	print_2d_array(ptr_out.get(), w, h);

	check_output_is_iota("copy from 2D texture into (managed) global memory", ptr_out.get(), arr.size());

	// copy between arrays and memory spaces
	cuda::array_t<float, 2> other_arr(device, dims);
	cuda::memory::copy(other_arr, ptr_out.get());
	cuda::memory::copy(ptr_in.get(), other_arr);

	check_output_is_iota("copy from (managed) global memory into a 2D array", ptr_in.get(), arr.size());
	
	// also asynchronously
	auto stream = device.create_stream(cuda::stream::async);
	cuda::memory::async::copy(other_arr, ptr_out.get(), stream);
	cuda::memory::async::copy(ptr_in.get(), other_arr, stream);
	device.synchronize();

	check_output_is_iota("copy from (managed) global memory into a 2D array, asynchronously", ptr_in.get(), arr.size());
}

int main()
{
	auto device = cuda::device::current::get();

	// array dimensions
	constexpr size_t w = 3;
	constexpr size_t h = 3;
	constexpr size_t d = 3;

	array_3d_example(device, w, h, d);
	array_2d_example(device, w, h);
	device.synchronize();
}
