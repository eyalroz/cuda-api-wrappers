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
#include <cuda/nvrtc.hpp>

#include "type_name.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <experimental/filesystem>



template <typename T>
bool are_close(T in, T out) {
  return fabs(in - out) <= 1e-5f * fabs(in);
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

bool try_compilation(
	const cuda::rtc::program_t &program,
	const cuda::device_t &device,
	cuda::rtc::compilation_options_t compilation_options = {})
{
	try {
		compilation_options.set_language_dialect("c++11");
		program.compile_for(device, compilation_options);
	}
	catch(std::exception& err) {
		std::cerr << "Program compilation failed: " << err.what() << '\n';
		auto compilation_log = program.compilation_log();
		std::cerr << "Compilation options: " << compilation_options << '\n';
		if (not compilation_log.empty()) {
			std::cerr
				<< "Compilation log:\n"
				<< string_view(compilation_log.data(), compilation_log.size()) << '\n'
				<< std::flush;
		}
		return false;
	}
	return true;
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
	auto program = cuda::rtc::program::create("my_program", source_with_instantiation.c_str());
	auto device = cuda::device::current::get();
	program.register_global(instantiation_name);
	if (not try_compilation(program, device)) { return false; }
	auto mangled_kernel_name = program.get_mangling_of(instantiation_name);
	auto module = cuda::module::create(device, program);
	// TODO: A kernel::get(const module_t& module, const char* mangled_name function)
	auto kernel = module.get_kernel(mangled_kernel_name);

	auto d_data = cuda::memory::device::make_unique<T>(device);
	T h_data = 5;
	cuda::memory::copy_single<T>(d_data.get(), &h_data);

	auto single_thread_launch_config = cuda::make_launch_config( cuda::grid::complete_dimensions_t::point());
	device.launch(kernel, single_thread_launch_config, d_data.get());
	cuda::memory::copy_single<T>(&h_data, d_data.get());
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
		// part of the CUDA API wrappers, but is actually both straightforward and flexible).
	std::string source_with_instantiation = append_kernel_instantiation(program_source, my_kernel2_instantiation_name);
	std::vector<std::pair<const char*, const char*>> headers = {
		{"example_headers/my_header4.cuh", my_header4_cuh_contents}
	};

	auto program = cuda::rtc::program::create("my_program1", source_with_instantiation.c_str(), headers);
	auto device = cuda::device::current::get();
	program.register_globals(kernel_names[0], my_kernel2_instantiation_name);
	cuda::rtc::compilation_options_t options;
	options.use_fast_math = true;
	options.default_execution_space_is_device = true;
		// This is necessary because the headers included by this program have functions without a
		// device/host qualification - and those default to being host functions, which NVRTC doesn't
		// compile.
	if (not try_compilation(program, device, options)) { return false; }
		// Note: Headers whose sources were not provided to the program on creation will be sought after
		// in the specified include directories (in our case, none), and the program's working directory.
	const char* mangled_kernel_names[2] = {
		program.get_mangling_of(kernel_names[0]),
		program.get_mangling_of(my_kernel2_instantiation_name)
	};
	auto module = cuda::module::create(device, program);
	auto my_kernel1 = module.get_kernel(mangled_kernel_names[0]);
	auto my_kernel2 = module.get_kernel(mangled_kernel_names[1]);

	auto indata = cuda::memory::device::make_unique<T>(device);
	auto outdata = cuda::memory::device::make_unique<T>(device);
	T inval = 3.14159f;
	cuda::memory::copy_single<T>(indata.get(), &inval);

	auto launch_config = cuda::make_launch_config(cuda::grid::complete_dimensions_t::point());
	cuda::launch(my_kernel1, launch_config, indata.get(), outdata.get());
	cuda::launch(my_kernel2, launch_config, indata.get(), outdata.get());

	T outval = 0;
	cuda::memory::copy_single(&outval, outdata.get());
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

	auto program = cuda::rtc::program::create("const_program", const_program_source);

	cuda::rtc::compilation_options_t options;
	options.use_fast_math = true;
	program.register_globals(names.kernel, names.a, names.b_a, names.c_b_a);
	auto device = cuda::device::current::get();
	if (not try_compilation(program, device, options)) { return false; }
	auto module = cuda::module::create(device, program);

	auto mangled_kernel_name = program.get_mangling_of(names.kernel);
	auto kernel = module.get_kernel(mangled_kernel_name);
	int inval[] = {2, 4, 8};
	auto a = module.get_global_region(program.get_mangling_of(names.a));
	auto b_a = module.get_global_region(program.get_mangling_of(names.b_a));
	auto c_b_a = module.get_global_region(program.get_mangling_of(names.c_b_a));
	cuda::memory::copy(a, &inval[0]);
	cuda::memory::copy(b_a, &inval[1]);
	cuda::memory::copy(c_b_a, &inval[2]);
	auto outdata = cuda::memory::device::make_unique<int[]>(device, n_const);
	auto launch_config = cuda::make_launch_config(cuda::grid::complete_dimensions_t::point());
	cuda::launch(kernel, launch_config, outdata.get());
	int outval[n_const];
	cuda::memory::copy(outval, outdata.get(), sizeof(outval));

	return std::equal(inval, inval + n_const, outval);
}

bool test_constant_2()
{
	// test __constant__ array look up in header nested in both anonymous and explicit namespace
	constexpr int n_const = 3;
	const char* second_kernel_name = "constant_test2";
	auto program = cuda::rtc::program::create_empty("const_program_2");
	cuda::rtc::compilation_options_t options;
	options.preinclude_files.emplace_back("example_headers/constant_header.cuh");
	options.use_fast_math = true;
	const char* name_of_anon_b_a = "&b::a";
	program.register_globals(second_kernel_name, name_of_anon_b_a);
	auto device = cuda::device::current::get();
	if (not try_compilation(program, device, options)) { return false; }
	auto module = cuda::module::create(device, program);
	auto anon_b_a = module.get_global_region(program.get_mangling_of(name_of_anon_b_a));
	auto kernel = module.get_kernel(program.get_mangling_of(second_kernel_name));
	int inval[] = {3, 5, 9};
	cuda::memory::copy(anon_b_a, inval);
	auto launch_config = cuda::make_launch_config(cuda::grid::complete_dimensions_t::point());
	auto outdata = cuda::memory::device::make_unique<int[]>(device, n_const);
	cuda::launch(kernel, launch_config, outdata.get());
	int outval[n_const];
	auto ptr = outdata.get();
	cuda::memory::copy(outval, ptr);
	return std::equal(inval, inval + n_const, outval);
}

int main(int, char**)
{
	namespace fs = std::experimental::filesystem;
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

