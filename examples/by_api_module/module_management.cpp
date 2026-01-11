/**
 * @file
 *
 * @brief An example program utilizing most/all calls
 * from the CUDA Driver API module: Module Management
 */
#include "../common.hpp"
#include "../string_view.hpp"
#include "../type_name.hpp"
#include "../zip.hpp"

#include <cuda/api.hpp>
#include <cuda/rtc.hpp>
#include <vector>

using std::cout;
using std::endl;
using nonstd::string_view;

constexpr const char *constant_name = "&a";
constexpr const char* basic_kernel_names[2] = { "my_kernel1", "my_kernel2" };

/*

__global__ void foo(int x)
{
	if (threadIdx.x == 0) {
		printf("Block %u is executing (with v = %d)\n", blockIdx.x, x);
	}
}

template <typename T>
__global__ void bar(T x)
{
	if (threadIdx.x == 0) {
		printf("Block %u is executing bar\n", blockIdx.x);
	}
}

*/

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
	(void) std::initializer_list<int>{((void)f(std::forward<Args>(args)), 0)...};
	return f;
}

template <typename... Ts>
std::string make_instantiation_name(string_view base_name, Ts&&... args)
{
	std::stringstream sstr;
	bool at_first_argument { true };
	(void) at_first_argument; // Avoiding some silly warnings by compiler which should know better.
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


std::pair<cuda::rtc::compilation_output_t<cuda::cuda_cpp>, std::vector<std::string>>
get_compiled_program(const cuda::device_t &device)
{
	const char* program_source =R"(


__constant__ int a;

__global__
void my_kernel1(float const* indata, float* outdata) {
    outdata[0] = indata[0] + 1;
    outdata[0] -= 1;
}

template<int C, typename T>
__global__
void my_kernel2(float const* indata, float* outdata) {
    for( int i=0; i<C; ++i ) {
        outdata[0] =-indata[0];
    }
};


)";

	enum { C = 123 };
	std::vector<std::string> instantiation_names;
	instantiation_names.push_back(basic_kernel_names[0]);
	instantiation_names.push_back(make_instantiation_name(basic_kernel_names[1], std::to_string(C), type_name<float>()));

	std::string source_with_instantiation = append_kernel_instantiation(program_source, instantiation_names[1]);
	auto program = cuda::rtc::program::create<cuda::cuda_cpp>("my_program1")
		.set_source(source_with_instantiation)
		.set_target(device)
		.add_registered_global(instantiation_names[0])
		.add_registered_global(instantiation_names[1])
		.add_registered_global(constant_name);
	program.options()
		.set_language_dialect("c++11")
		.default_execution_space_is_device = true;// Note: Headers whose sources were not provided to the program on creation will be sought after
	auto compilation_result = program.compile();
	if (not compilation_result.succeeded()) {
		handle_compilation_failure(compilation_result, program.options());
		throw std::runtime_error("Program compilation failed.");
	}
	return make_pair(std::move(compilation_result), instantiation_names);
}

bool basic_module_tests(
	const char* title,
	const cuda::device_t &device,
	const cuda::rtc::compilation_output_t<cuda::cuda_cpp> &compilation_result,
	const char *const *mangled_kernel_names,
	const cuda::module_t &module
#if CUDA_VERSION >= 12040
	, cuda::unique_span<cuda::kernel_t> &module_kernels
#endif
	)
{
	std::cout << "\nRunning basic module tests for " << title << ":\n";
	bool test_result { true };
#if CUDA_VERSION >= 12040
	auto module_kernels_ = module.get_kernels();
	module_kernels = std::move(module_kernels_);
#endif

	test_result = test_result and (module.device_id() == device.id());
	test_result = test_result and (module.device() == device);
	test_result = test_result and (module.context() == device.primary_context(cuda::do_not_hold_primary_context_refcount_unit));
	test_result = test_result and (module.context_handle() == cuda::device::primary_context::detail_::get_handle(device.id()));

	{
		auto a = module.get_global_region(compilation_result.get_mangling_of(constant_name));
		static constexpr const size_t size_of_int_on_cuda_devices{4};
		test_result = test_result and (a.size() == size_of_int_on_cuda_devices);
		test_result = test_result and (a.start() != nullptr);
	}

	auto my_kernel1 = module.get_kernel(mangled_kernel_names[0]);
	auto my_kernel2 = module.get_kernel(mangled_kernel_names[1]);

	auto list_kernel =
		[](const char * title, const char * mangled_name, cuda::optional<const char*> unmangled) {
			std::cout
				<< title << ":\n"
				<< "  unmangled: " << unmangled.value_or("N/A") << '\n'
				<< "  mangled:   " << mangled_name << "\n"
#if __GNUC__
				<< "  demangled: " << demangle(mangled_name) << '\n'
#endif
				;
		};
	list_kernel("First kernel", mangled_kernel_names[0], basic_kernel_names[0]);
	list_kernel("Second kernel", mangled_kernel_names[1], basic_kernel_names[1]);
	std::cout  << '\n';
#if CUDA_VERSION >= 12040
	{
		auto num_kernels = module.get_num_kernels();
		std::cout << "Module has " << num_kernels << " kernels.\n";
		test_result = test_result and (num_kernels == 2);
		test_result = test_result and (module_kernels.size() == num_kernels);

		if (module_kernels.size() == 2) {
			list_kernel("First enumerated kernel", module_kernels[0].mangled_name(), {});
			list_kernel("Second enumerated kernel", module_kernels[1].mangled_name(), {});
			test_result = test_result and
				((module_kernels[0] == my_kernel1 and module_kernels[1] == my_kernel2)
				or (module_kernels[1] == my_kernel1 and module_kernels[0] == my_kernel2));
		}
	}
#endif // CUDA_VERSION >= 12040
	return test_result;
}

int main(int, char**)
{
	bool test_result { true };
	auto device = cuda::device::current::get();

	auto pair = get_compiled_program(device);
	auto& compilation_result = pair.first;
	auto& kernel_names = pair.second;

	const char* mangled_kernel_names[2] = {
		compilation_result.get_mangling_of(kernel_names[0]),
		compilation_result.get_mangling_of(kernel_names[1])
	};

	auto module = cuda::module::create(device, compilation_result);
#if CUDA_VERSION >= 12040
	cuda::unique_span<cuda::kernel_t> module_kernels;
#endif
	test_result = test_result and basic_module_tests(
		"module created from compiled program",
		device, compilation_result, mangled_kernel_names, module
#if CUDA_VERSION >= 12040
		, module_kernels
#endif
		);

	cuda::link::options_t link_opts;
	link_opts.default_load_caching_mode() =  cuda::caching_mode_t<cuda::memory_operation_t::load>::dont_cache;
	link_opts.generate_source_line_info = true;
	auto module2 = cuda::module::create(device, compilation_result, link_opts);
#if CUDA_VERSION >= 12040
	auto module_2_kernels = module2.get_kernels();
	test_result = test_result and module_2_kernels.size() == module_kernels.size();
	if (test_result) {
		for(size_t i = 0; i < module_kernels.size(); i++) {
			test_result = test_result and (module_kernels[i].mangled_name() == module_2_kernels[i].mangled_name());
		}
	}
#endif
	module2 = std::move(module);
	test_result = test_result and basic_module_tests(
		"move-assigned module",
		device, compilation_result, mangled_kernel_names, module2
#if CUDA_VERSION >= 12040
		, module_kernels
#endif
		);
	auto module3 { std::move(module2) };
	test_result = test_result and basic_module_tests(
		"move-constructed module",
		device, compilation_result, mangled_kernel_names, module3
#if CUDA_VERSION >= 12040
		, module_kernels
#endif
		);

	std::cout << '\n' << (test_result ? "SUCCESS" : "FAILURE") << '\n';
	exit(test_result ? EXIT_SUCCESS : EXIT_FAILURE);
}
