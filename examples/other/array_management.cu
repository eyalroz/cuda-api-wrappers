#include "../common.hpp"

#include <iostream>
#include <iomanip>
#include <numeric>

using cuda::size_t;

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
	const auto gtid_serialized = gtid.x + gtid.y * static_cast<unsigned>(w);

	if (gtid.x < w && gtid.y < h) {
		const float x = tex2D<float>(texture_source, gtid.x, gtid.y);
//		printf("Thread %u %u, reading value %.4g, and writing to index %3u\n", gtid.x, gtid.y, x, gtid_serialized);
		destination[gtid_serialized] = x;
	}
}

} // namespace kernels

template <typename T>
void check_output_is_iota(std::string name, cuda::span<T> actual) noexcept
{
	bool failed { false };
	for (size_t i = 0; i < actual.size(); ++i) {
		if (actual[i] != i) {
			if (not failed) {
				std::cerr << name << ": Output does not matched expected values:\n";
				failed = true;
			}
			std::cerr << "output[" << std::setw(3) << i << "] = " << actual[i] << " != " << i << '\n';
		}
	}
	if (failed) { std::cerr << '\n'; }
}

void array_3d_example(cuda::device_t& device, size_t w, size_t h, size_t d) {
	namespace grid = cuda::grid;

	const cuda::array::dimensions_t<3> dims = {w, h, d};
	auto arr = cuda::array::create<float>(device, dims);
	assert_(arr.device() == device);
	auto span_in = cuda::memory::managed::make_unique_span<float>(arr.size());
	std::iota(span_in.begin(), span_in.end(), (float) 0.0);
	auto span_out = cuda::memory::managed::make_unique_span<float>(arr.size());
	cuda::memory::copy(arr, span_in.get());
	cuda::texture_view tv(arr);
    assert_(tv.device() == device);
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

	auto launch_config = cuda::launch_configuration_t{grid_dims, block_dims};
	cuda::launch(
		kernels::from_3D_texture_to_memory_space,
		launch_config,
		tv.raw_handle(), span_out.data(), w, h, d);
	device.synchronize();
	check_output_is_iota("copy from 3D texture into (managed) global memory", span_out.get());

	// copy between arrays and memory spaces
	auto other_arr = cuda::array::create<float>(device, dims);
	cuda::memory::copy(other_arr, span_out.get());
	cuda::memory::copy(span_in, other_arr);

	check_output_is_iota("copy from (managed) global memory into a 3D array", span_in.get());

	// also asynchronously
	auto stream = device.create_stream(cuda::stream::async);
	cuda::memory::async::copy(other_arr, span_out, stream);
	cuda::memory::async::copy(span_in, other_arr, stream);
	device.synchronize();
	check_output_is_iota("copy from (managed) global memory into a 3D array, asynchronously", span_in);
}

template <typename T>
void print_2d_array(const char* title, cuda::span<T> a, size_t width, size_t height)
{
    std::cout << title << ":\n";
	for (size_t i = 0; i < height; ++i) {
		for (size_t j = 0; j < width; ++j) {
			std::cout << a[j + i * width] << ' ';
		}
		std::cout << '\n';
	}
}

void array_2d_example(cuda::device_t& device, size_t w, size_t h)
{
	const cuda::array::dimensions_t<2> dims = {w, h};
	auto arr = cuda::array::create<float>(device , dims);
	auto span_in = cuda::memory::managed::make_unique_span<float>(arr.size());
	std::iota(span_in.begin(), span_in.end(), (float) 0);

	std::cout << std::endl;

    print_2d_array("Data in span_in after initialization", span_in, w, h);

	cuda::memory::copy(arr, span_in);
	cuda::texture_view tv(arr);

	constexpr cuda::grid::block_dimension_t block_dim = 10;
	assert(div_rounding_up(w, block_dim) <= std::numeric_limits<cuda::grid::dimension_t>::max());
	assert(div_rounding_up(h, block_dim) <= std::numeric_limits<cuda::grid::dimension_t>::max());
	auto launch_config = cuda::launch_config_builder()
		.overall_dimensions(w, h)
		.block_dimensions(block_dim, block_dim)
		.build();

    auto span_out = cuda::memory::managed::make_unique_span<float>(arr.size());
    // The following is to make it easier to notice if nothing get copied
    // to the output
    std::iota(span_out.begin(), span_out.end() + arr.size(), (float) 90);
//    print_2d_array("Data at span_out after initialization", span_out.get(), w, h);

	cuda::launch(
		kernels::from_2D_texture_to_memory_space,
		launch_config,
		tv.raw_handle(), span_out.data(), w, h);
	cuda::memory::copy(span_out, arr);
	device.synchronize();
	print_2d_array("Data at span_out after execution of 'from_2D_texture_to_memory_space'", span_out, w, h);

	check_output_is_iota("copy from 2D texture into (managed) global memory", span_out.get());

	// copy between arrays and memory spaces
	auto other_arr = cuda::array::create<float>(device, dims);
	cuda::memory::copy(other_arr, span_out);
	cuda::memory::copy(span_in, other_arr);

	check_output_is_iota("copy from (managed) global memory into a 2D array", span_in);

	// also asynchronously
	auto stream = cuda::stream::create(device, cuda::stream::async);
	cuda::memory::async::copy(other_arr, span_out, stream);
	cuda::memory::async::copy(span_in, other_arr, stream);
	device.synchronize();

	check_output_is_iota("copy from (managed) global memory into a 2D array, asynchronously", span_in);
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

	std::cout << "\nSUCCESS\n";
}
