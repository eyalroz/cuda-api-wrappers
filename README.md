# Thin C++-flavored wrappers for the CUDA APIs:<br> Runtime, Driver, NVRTC, PTX compiler and NVTX

<!--Branch Build Status: Master [![Master Build Status](https://api.travis-ci.com/eyalroz/cuda-api-wrappers.svg?branch=master)](https://travis-ci.com/eyalroz/cuda-api-wrappers) | Development: [![Development Build Status](https://api.travis-ci.com/eyalroz/cuda-api-wrappers.svg?branch=development)](https://travis-ci.com/eyalroz/cuda-api-wrappers) -->


| Table of contents |
|:------------------|
|<sub>[General description](#general-description)<br>  - [Key features](#key-features)<br>[Motivation](#motivation)<br>[Detailed documentation](#detailed-documentation)<br>[Requirements](#requirements)<br>[Using the library in your project](#using-the-library-in-your-project)<br>[Coverage of the APIs](#coverage-of-the-apis)<br>[A taste of some features in play](#a-taste-of-some-features-in-play)<br>[Example programs](#example-programs)<br>  - [Modified CUDA samples](#modified-cuda-samples)<br>  - ['Coverage' programs - by API module](#coverage-programs---by-api-module)<br>[Want to help? Report a bug? Give feedback?](#want-to-help-report-a-bug-give-feedback)</sub>|


## General description

This is a header-only library of integrated wrappers around the core parts of NVIDIA's [CUDA](https://developer.nvidia.com/cuda-zone) execution ecosystem:

* The lower-level [CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
* The slightly higher-level CUDA [Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
* NVIDIA's dynamic CUDA code compilation library, [NVRTC](http://docs.nvidia.com/cuda/nvrtc/index.html)
* NVIDIA's out-of-driver, full-featured [PTX compiler library](https://docs.nvidia.com/cuda/ptx-compiler-api/index.html) (available since CUDA 11.1)
* The NVIDIA profiler in-program API, also known as [NVTX](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx) (the NVIDIA Toolkit Extensions library).

It is intended for those who would otherwise use these APIs directly, to make working with them be more intuitive and consistent, making use of modern C++ language capabilities, programming idioms and best practices. In a nutshell - making CUDA API work more fun :-)

Also, and importantly - while the wrappers seem to be "high-level", more "abstract" code - they are nothing more than a modern-C++-aesthetic arrangement of NVIDIA's own APIs. The wrapper library does not force any abstractions above CUDA's own, nor conventions regarding where to place data, how and when to perform synchronization, etc.; you have the complete range of expression of the underlying APIs.


### Key features

In contrast to the above, this library provides:

- **Seamlessly integrated functionality** of the Driver, Runtime and NVRTC API (NVTX doesn't integrate all that much, seamlessly or otherwise, but it's readily usable).
- All functions and methods throw **exceptions** on failure, which carry status information; no need to check return values.
- Methods and **functions return what they produce**, since they don't need to return a status code. No more having to pre-allocate result variables and pass pointers to them as out-parameters. Better compositionality!
- Judicious **namespacing** (and some internal namespace-like classes) for clarity and for semantic grouping of related functionality.
- There are **proxy/wrapper objects** for devices, streams, events, kernels, contexts, modules, link processes, timed intervals and so on - all using the [CADRe/RAII](http://en.cppreference.com/w/cpp/language/raii) convention; you don't have to remember to free or release your resources yourself.
- You can **forget about numeric IDs and handles**; the proxy classes will fit everywhere. Of course, you can still get those numeric values for cooperation with other CUDA-related software.
- Various [Plain Old Data](http://en.cppreference.com/w/cpp/concept/PODType) structs adorned with **convenience methods and operators** (e.g. device properties, block and grid dimensions).
- Aims for **clarity and straightforwardness** in naming and semantics, so that you don't need to refer to the official documentation to understand what each class and function do.
- Aims for conformance with the [C++ core guidelines](https://github.com/isocpp/CppCoreGuidelines).
- **Header-only**: No need to compile anything special to use the library.
- Thin and **lightweight**:
    - No work done behind your back, no caches or indices or any such thing - with almost no exceptions. (Some do exist, though: Ensuring Runtime-API and Driver-API compatibility sometimes requires it; and there is also the matter of creating devices' primary context).
    - No costly inheritance structure, vtables, virtual methods and so on, for almost all wrappers; they vanishes almost entirely on compilation.
    - All "Runtime-API level" actions are implemented so as not to disrupt "Driver-API-level" work.
- Permissive free software license: [3-BSD](https://github.com/eyalroz/cuda-api-wrappers/blob/master/LICENSE).

There is one noteworthy caveat: The wrapper API calls cannot make assumptions about previous or later code of yours, which means some of them require more calls to obtain the current context handle or push a(n existing) context, then pop it. While these calls are cheap, they are still non-trivial and can't be optimized away.

## Motivation

NVIDIA provides two main APIs for using [CUDA](https://developer.nvidia.com/cuda-zone): The Runtime API and the
Driver API. These suffer from several deficiencies:

* They are both C-style APIs, targeting the lowest common denominator of language facilities for abstraction, safety and ease of use.
* ... although the Runtime API has a few [exceptions](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html), so you actually need to write C++ to use it.
* The runtime API is supposedly the higher-level, more convenient version of the Driver API; in fact, it's missing a lot of important functionality, and doesn't can't be used in conjuction with other CUDA facilities, like the NVRTC dynamic compilation library. Consequently, you're forced to use _both_ the runtime and the driver API.
* It is difficult to use both the runtime and driver API in conjuction; if you're not careful, you'll mess up things like the context stack.
* The runtime API makes multiple assumptions which you might not want it to make.
* You have to manually check every call to every API function, everywhere.
* You have to work with pointers a lot, and pointers-to-pointers, since the API functions mostly return status codes rather than outputs; this will also prevent you from composing.
* You will need to remember to release all resources you allocate or create - or else bad things happen.
* There is a very large number of API functions, many of which are related, have similar names, and are easy to confuse.
* You will be working with a lot of numeric and pointer-based handles, instead of class or structs. Other than the poor aesthetics, this makes it easy to mix up resource-handle and pure-number parameters to functions.

You may have noticed this list reads like the opposite of the [key features](#key-features), listed above: The idea is to make this library overcome and rectify all of these deficiencies as much as possible.


## Detailed documentation

Detailed Doxygen-genereated documentation is [available](https://codedocs.xyz/eyalroz/cuda-api-wrappers/). It is mostly complete for the Runtime API wrappers, less so for the rest of the wrappers.

## Requirements

- CUDA: v11.x or later recommended, v9.0 or later supported.

    Remember that an NVIDIA driver compatible with your CUDA version also needs to be installed. Typically, this can be the one bundled in your CUDA distribution itself.

- Other software:
     - A C++11-capable compiler compatible with your version of CUDA.
     - CMake v3.25 or later; it's very easy to [download and install](https://cmake.org/download/) a recent version - no need to build it yourself.

- An NVIDIA GPU supporting Unified Virtual Addressing (UVA), i.e. Fermi microarchitecture or later. With earlier GPUs, memory copying, and other functionality relying on automtically determining where a memory address is located, will fail.

## Using the library in your project

### Projects using CMake

For CMake, you have three alternatives for obtaining the library to use in your project:

1. (apriori) Manually download a release tarball from the [Releases](https://github.com/eyalroz/cuda-api-wrappers/releases) page. Then, configure it with CMake, build it, and install it - to a place visible to cmake when it searches for packages (see [CMAKE_PREFIX_PATH](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html)).
3. (apriori) Use the [conan](https://conan.io) package management tool: Follow the instructions in the Conan tutorial on [Building a simple CMake project using Conan](https://docs.conan.io/2/tutorial/consuming_packages/build_simple_cmake_project.html), but instead of `zlib`, use `cuda-api-wrappers` with the version you're interested in. Briefly, you will need to create an appropriate `conanfile.txt` file; make sure you have a Conan profile; and use `conan install` to target your project's build directory. This will create a [CMake toolchain file](https://cmake.org/cmake/help/latest/variable/CMAKE_TOOLCHAIN_FILE.html) with which you actually build your project.
2. (apriori) Use the [vcpkg](https://github.com/microsoft/vcpkg) package management tool:
   ```
   vcpkg install cuda-api-wrappers
   ```
3. (at config time) use CMake's `FetchContent` module to have CMake itself obtain the project source code and make it part of your own project's build, e.g.:
   ```
   include(FetchContent)
   FetchContent_Declare(cuda-api-wrappers_library
       GIT_REPOSITORY https://github.com/eyalroz/cuda-api-wrappers.git
       GIT_TAG v12.34.56 # Replace this with a real available version
       OVERRIDE_FIND_PACKAGE
   )
   ```
Now that you have the package, in your project's `CMakeLists.txt`, you write:
```
find_package(cuda-api-wrappers CONFIG REQUIRED)
```
This will let you use three targets within the `cuda-api-wrappers::` namespace: `runtime-and-driver`, `nvrtc` and `nvtx`. For example:
```
target_link_library(my_app cuda-api-wrappers::runtime-and-driver)
```
**Use not involving CMake:**

Since this is a header-only library, you can simply add the `src/` subdirectory as one of your project's include directories. However, if you do this, it will be up to you to make sure and have the CUDA include directory in you include path as well, and to link against the CUDA driver, runtime API, NVRTC and/or NVTX libraries as appropriate.

## Coverage of the APIs

Most, but not quite all, API calls in the Runtime, Driver, NVTX and NVRTC are covered by these wrappers. You can find the main omissions as [issues tagged with "missing-cuda-feature"](https://github.com/eyalroz/cuda-api-wrappers/issues?q=is%3Aissue+is%3Aopen+label%3Amissing-cuda-feature), intended for further development work. Additionally, the wrapper library does not cover the APIs for interoperability with OpenGL, Direct3D, EGL and VDAPU.

The [Milestones](https://github.com/eyalroz/cuda-api-wrappers/milestones) indicates some features which aren't covered and are slated for future work. Since I am not currently working on anything graphics-related, there are no short-term plans to extend coverage to more graphics-related APIs; however - PRs are welcome.

## A taste of some features in play

Don't you wish you were able to type in, say:

	auto callback =	[&foo] { std::cout << "Hello " << foo << " world!\n"; }
	my_stream.enqueue.host_invokable(callback);

... and have that just work? Well, now it does!

To be a little more thorough than just an anecdote, let's relate back to some of the design principles listed above:

#### Use of namespaces (and internal classes)
With this library, you would do `cuda::memory::host::allocate(num_bytes)` instead of calling `cudaMallocHost()` or `cuMemAllocHost()`; and if you want to allocate global device memory, it would be `my_device.memory.allocate(num_bytes)` (or `cuda::device::get(device_id).memory.allocate(num_bytes)`. Or it could be a stand-alone function as well, i.e. `cuda::memory::device::allocate(my_device, num_bytes)`. The compiled version of these calls will be nothing but the sequence of API calls: `cuInit()`, `cuDevicePrimaryCtxRetain()`, `cuCtxPushCurrent()`, `cuMemAlloc()` etc.

#### Adorning POD structs with convenience methods
The expression
```
my_device.compute_capability() >= cuda::make_compute_capability(60)
```
is a valid comparison, true for all devices with a Pascal-or-later micro-architecture. This, despite the fact that `struct cuda::compute_capability_t` is a POD type with two unsigned integer fields, not a scalar.

#### Meaningful naming
Instead of using
```
cudaError_t cudaEventCreateWithFlags(
    cudaEvent_t* event,
    unsigned int flags)
```
which requires you remember what you need to specify as flags and how, you create a `cuda::event_t` proxy object, using the function:
```
cuda::event_t cuda::event::create(
    cuda::device_t  device,
    bool            uses_blocking_sync,
    bool            records_timing      = cuda::event::do_record_timing,
    bool            interprocess        = cuda::event::not_interprocess)
```
The default values here are `enum : bool`'s, which you can use yourself when creating non-default-parameter events - to make the call more easily readable than with `true` or `false`.

## Example programs

In lieu of a full-fledged user's guide, I'm providing several kinds of example programs; browsing their source you'll know most of what there is to know about the API wrappers. To build and run the examples (just as a sanity check), execute the following (in a Unix-style command shell):

    cmake -S . -B build -DCAW_BUILD_EXAMPLES=ON .
    cmake --build build/
    find build/examples/bin -type f -executable -exec "{}" ";"

The two main kinds of example programs are:

#### Modified CUDA samples

The CUDA distribution contains sample programs demostrating various features and concepts. A few of these - which are not focused on device-side work - have been adapted to use the API wrappers - completely foregoing direct use of the CUDA Runtime API itself. You will find them in the [modified CUDA samples](https://github.com/eyalroz/cuda-api-wrappers/tree/master/examples/modified_cuda_samples/) example programs folder.

#### 'Coverage' programs - by API module

Gradually, an example program is being added for each one of the CUDA Runtime API [Modules](http://docs.nvidia.com/cuda/cuda-runtime-api/modules.html#modules), in which the approach replacing use of those module API calls by use of the API wrappers is demonstrated. These per-module example programs can be found [here](https://github.com/eyalroz/cuda-api-wrappers/tree/master/examples/by_runtime_api_module/).

## Want to help? Report a bug? Give feedback?

* Noticed a bug, compatibility issue, missing functionality or other problem? Please [file the issue](https://github.com/eyalroz/cuda-api-wrappers/issues) here on GitHub.
* Started using the library in a publicly-available project? Please email [@eyalroz](https://github.com/eyalroz).
* Want to help test new, improved versions? Please email [@eyalroz](https://github.com/eyalroz).
* Interested in collaborating on coding/research work related to the library? Have a look at [@eyalroz' website](https://github.com/eyalroz).
