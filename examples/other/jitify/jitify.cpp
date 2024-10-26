/**
 * @file
 *
 * @brief An adaptation of NVIDIA's jitify library's "simple example"
 * program to use the CUDA API wrappers.
 *
 * @copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 * @copyright (c) 2022, Eyal Rozenberg <eyalroz1@gmx.com>
 *
 * @license
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of NVIDIA CORPORATION nor the names of its
 *   contributors nor the name of any of the other copyright holders
 *   above may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "../../common.hpp"

#include <cuda/api.hpp>
#include <cuda/rtc.hpp>

#include "../../type_name.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#if __cplusplus >= 201703
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

template <typename T>
bool are_close(T in, T out) {
  return fabs(in - out) <= 1e-5 * fabs(in);
}

/**
 * Takes a program and appends to it an instantiation of a templated function,
 * for the specified name of the instantiation.
 *
 * @param program_source CUDA C++ source code with a templated function
 * @param instantiation_name As in C++ in general, e.g. for a template "foo<typename T, int V>",
 * a possible value could be "foo<int, 123>".
 * @return the program with instantiation statement appended to its end.
 */
std::string append_kernel_instantiation(string_view program_source, string_view instantiation_name)
{
	std::stringstream sstr;

	sstr << program_source
	     << "\n"
	     << "template __global__ decltype(" << instantiation_name << ") " << instantiation_name << ";\n";
	return sstr.str();
}

template<class F, class...Args>
F for_each_arg(F f, Args&&...args) {
	std::initializer_list<int>{((void)f(std::forward<Args>(args)), 0)...};
	return f;
}

template <typename... Ts>
std::string make_instantiation_name(string_view base_name, Ts&&... args)
{
	std::stringstream sstr;
	bool at_first_argument { true };
	sstr << base_name << '<';
	for_each_arg(
		[&](string_view s) {
			if (at_first_argument) {
				at_first_argument = false;
			}
			else { sstr << ", "; }
			sstr << s;
		}, args...);
	sstr << '>';
	return sstr.str();
}
/**
 *
 * @param compilation_output
 * @param fine_day hello world
 */
void handle_compilation_failure(
	const cuda::rtc::compilation_output_t<cuda::cuda_cpp>& compilation_output,
	cuda::rtc::compilation_options_t<cuda::cuda_cpp> compilation_options = {})
{
	std::cerr << "Program compilation failed:\n";
	auto compilation_log = compilation_output.log();
	std::cerr << "Compilation options were: " << cuda::rtc::render(compilation_options) << '\n';
	if (not compilation_log.empty()) {
		std::cerr
			<< "Compilation log:\n"
			<< string_view(compilation_log.data(), compilation_log.size()) << '\n'
			<< std::flush;
	}
}

template <typename T>
bool test_simple() {
	const auto kernel_name = "my_kernel";
  	const char* program_source =
R"(template<int N, typename T>
__global__
void my_kernel(T* data) {
    T data0 = data[0];
    for( int i=0; i<N-1; ++i ) {
        data[0] *= data0;
    }
};
)";
	auto instantiation_name = make_instantiation_name(kernel_name, std::to_string(3), type_name<T>());
	std::string source_with_instantiation = append_kernel_instantiation(program_source, instantiation_name);
	auto device = cuda::device::current::get();
	auto program = cuda::rtc::program_t<cuda::cuda_cpp>("my_program")
		.set_source(source_with_instantiation)
		.set_target(device)
		.add_registered_global(instantiation_name);
	program.options().set_language_dialect("c++11");
	auto compilation_result = program.compile();

	if (not compilation_result.succeeded()) {
		handle_compilation_failure(compilation_result, program.options());
	}
	auto mangled_kernel_name = compilation_result.get_mangling_of(instantiation_name);
	auto module = cuda::module::create(device, compilation_result);
	// TODO: A kernel::get(const module_t& module, const char* mangled_name function)
	auto kernel = module.get_kernel(mangled_kernel_name);

	auto d_data = cuda::memory::make_unique_span<T>(device, 1);
	T h_data = 5;
	cuda::memory::copy_single<T>(d_data.data(), &h_data);

	auto single_thread_launch_config = cuda::launch_configuration_t(cuda::grid::composite_dimensions_t::point());
	device.launch(kernel, single_thread_launch_config, d_data.get());
	cuda::memory::copy_single<T>(&h_data, d_data.data());
	return are_close(h_data, 125.f);
}

template <typename T>
bool test_kernels() {
	const char* my_header4_cuh_contents =
R"(#pragma once
template<typename T>
T pointless_func(T x) {
	return x;
}
#warning "Here!"
)";

  	const char* program_source =
R"(
#include "example_headers/my_header1.cuh"
#include "example_headers/my_header2.cuh"
#include "example_headers/my_header3.cuh"
#include "example_headers/my_header4.cuh"

__global__
void my_kernel1(float const* indata, float* outdata) {
    outdata[0] = indata[0] + 1;
    outdata[0] -= 1;
}

template<int C, typename T>
__global__
void my_kernel2(float const* indata, float* outdata) {
    for( int i=0; i<C; ++i ) {
        outdata[0] = pointless_func(identity(sqrt(square(negate(indata[0])))));
    }
};
)";
  	const char* kernel_names[2] = { "my_kernel1", "my_kernel2" };

	enum { C = 123 };
	auto my_kernel2_instantiation_name = make_instantiation_name(kernel_names[1], std::to_string(C), type_name<T>());
		// Note: In the original jitify.cpp function, there were 6 different equivalent ways to trigger an
		// instantiation of the template. I don't see why that's at all useful, but regardless - here we
		// instantiate by simply printing whatever is passed to the instantiation function (which is not
		// part of the CUDA API wgetrappers, but is actually both straightforward and flexible).
	std::string source_with_instantiation = append_kernel_instantiation(program_source, my_kernel2_instantiation_name);
	std::vector<std::pair<const char*, const char*>> headers = {
		{"example_headers/my_header4.cuh", my_header4_cuh_contents }
	};

	auto device = cuda::device::current::get();
	auto program = cuda::rtc::program::create<cuda::cuda_cpp>("my_program1")
		.set_source(source_with_instantiation)
		.set_headers(headers)
		.set_target(device)
		.add_registered_global(kernel_names[0])
		.add_registered_global(my_kernel2_instantiation_name);
	auto& options = program.options();
	options.set_language_dialect("c++11");
	options.use_fast_math = true;
	options.default_execution_space_is_device = true;
		// This is necessary because the headers included by this program have functions without a
		// device/host qualification - and those default to being host functions, which NVRTC doesn't
		// compile.

	auto compilation_result = program.compile();
	// Note: Headers whose sources were not provided to the program on creation will be sought after
	// in the specified include directories (in our case, none), and the program's working directory.

	if (not compilation_result.succeeded()) {
		handle_compilation_failure(compilation_result, program.options());
	}

	const char* mangled_kernel_names[2] = {
		compilation_result.get_mangling_of(kernel_names[0]),
		compilation_result.get_mangling_of(my_kernel2_instantiation_name)
	};
	auto module = cuda::module::create(device, compilation_result);
	auto my_kernel1 = module.get_kernel(mangled_kernel_names[0]);
	auto my_kernel2 = module.get_kernel(mangled_kernel_names[1]);

	auto indata = cuda::memory::make_unique_span<T>(device, 1);
	auto outdata = cuda::memory::make_unique_span<T>(device, 1);
	T inval = 3.14159f;
	cuda::memory::copy_single<T>(indata.data(), &inval);

	auto launch_config = cuda::launch_configuration_t(cuda::grid::composite_dimensions_t::point());
	cuda::launch(my_kernel1, launch_config, indata.get(), outdata.get());
	cuda::launch(my_kernel2, launch_config, indata.get(), outdata.get());

	T outval = 0;
	cuda::memory::copy_single(&outval, outdata.data());
	// std::cout << inval << " -> " << outval << std::endl;
	return are_close(inval, outval);
}

bool test_constant()
{
	constexpr int n_const = 3;

	const char *const_program_source =
R"(#pragma once

__constant__ int a;
namespace b { __constant__ int a; }
namespace c { namespace b { __constant__ int a; } }

__global__ void constant_test(int *x) {
  x[0] = a;
  x[1] = b::a;
  x[2] = c::b::a;
}
)";
	struct {
		const char *kernel = "constant_test";
		const char *a = "&a";
		const char *b_a = "&b::a";
		const char *c_b_a = "&c::b::a";
	} names;

	auto device = cuda::device::current::get();
	auto program = cuda::rtc::program::create<cuda::cuda_cpp>("const_program")
		.set_source(const_program_source)
		.add_registered_global(names.kernel)
		.add_registered_global(names.a)
		.add_registered_global(names.b_a)
		.add_registered_global(names.c_b_a)
		.set_target(device);

	auto& options = program.options();
	options.set_language_dialect("c++11");
	options.use_fast_math = true;
	auto compilation_result = program.compile();
	if (not compilation_result.succeeded()) {
		handle_compilation_failure(compilation_result, program.options());
	}
	auto module = cuda::module::create(device, compilation_result);

	auto mangled_kernel_name = compilation_result.get_mangling_of(names.kernel);
	auto kernel = module.get_kernel(mangled_kernel_name);
	int inval[] = {2, 4, 8};
	auto a = module.get_global_region(compilation_result.get_mangling_of(names.a));
	auto b_a = module.get_global_region(compilation_result.get_mangling_of(names.b_a));
	auto c_b_a = module.get_global_region(compilation_result.get_mangling_of(names.c_b_a));
	cuda::memory::copy_2(a, &inval[0]);
	cuda::memory::copy_2(b_a, &inval[1]);
	cuda::memory::copy_2(c_b_a, &inval[2]);
	auto outdata = cuda::memory::make_unique_span<int>(device, n_const);
	auto launch_config = cuda::launch_configuration_t(cuda::grid::composite_dimensions_t::point());
	cuda::launch(kernel, launch_config, outdata.data());
	int outval[n_const];
	cuda::memory::copy_2(outval, outdata.get(), sizeof(outval));

	return std::equal(inval, inval + n_const, outval);
}


bool test_constant_2()
{
	// test __constant__ array look up in header nested in both anonymous and explicit namespace
	constexpr int n_const = 3;
	const char* second_kernel_name = "constant_test2";
	auto device = cuda::device::current::get();
	const char* name_of_anon_b_a = "&b::a";
	auto program = cuda::rtc::program::create<cuda::cuda_cpp>("const_program_2")
		.add_registered_global(second_kernel_name)
		.add_registered_global(name_of_anon_b_a)
		.set_target(device);
	auto& options = program.options();
	options.preinclude_files.emplace_back("example_headers/constant_header.cuh");
	options.use_fast_math = true;
	auto compilation_result = program.compile();
	if (not compilation_result.succeeded()) {
		handle_compilation_failure(compilation_result, program.options());
	}
	auto module = cuda::module::create(device, compilation_result);
	auto anon_b_a = module.get_global_region(compilation_result.get_mangling_of(name_of_anon_b_a));
	auto kernel = module.get_kernel(compilation_result.get_mangling_of(second_kernel_name));
	int inval[] = {3, 5, 9};
	cuda::memory::copy_2(anon_b_a, inval);
	auto launch_config = cuda::launch_configuration_t(cuda::grid::composite_dimensions_t::point());
	auto outdata = cuda::memory::make_unique_span<int>(device, n_const);
	cuda::launch(kernel, launch_config, outdata.data());
	int outval[n_const];
	auto ptr = outdata.get();
	cuda::memory::copy_2(outval, ptr);
	return std::equal(inval, inval + n_const, outval);
}

int main(int, char**)
{
	fs::path extra_headers_dir { "example_headers" };
	fs::exists(extra_headers_dir) or die_(
		"Cannot run the jitify test without the extra headers directory " + extra_headers_dir.string());
	fs::is_directory(extra_headers_dir) or die_(extra_headers_dir.string() + " is not a directory");

	bool test_simple_result = test_simple<float>();
	bool test_kernels_result = test_kernels<float>();
#if CUDA_VERSION >= 10000
	bool test_constant_result = test_constant();
	bool test_constant_2_result = test_constant_2();
#endif // CUDA_VERSION >= 10000

	// Note: There's no source-based or signature-based kernel caching mechanism - but
	// you can certainly keep the modules and kernels built within the test_XXXX functions
	// alive and launch the kernels again whenever you like.

	using std::cout;
	using std::endl;

	auto pass_or_fail = [](bool result) noexcept { return result ? "PASSED" : "FAILED"; };

	cout << "test_simple<float>:              " << pass_or_fail(test_simple_result) << endl;
	cout << "test_kernels<float>:             " << pass_or_fail(test_kernels_result) << endl;
#if CUDA_VERSION >= 10000
	cout << "test_constant:                   " << pass_or_fail(test_constant_result) << endl;
	cout << "test_constant_2:                 " << pass_or_fail(test_constant_2_result) << endl;
#endif // CUDA_VERSION >= 10000

	return not(
		test_simple_result
		and test_kernels_result
#if CUDA_VERSION >= 10000
		and test_constant_result
		and test_constant_2_result
#endif // CUDA_VERSION >= 10000
	);
}

