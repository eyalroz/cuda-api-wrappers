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
- You can  **forget about numeric IDs andhandles**; the proxy classes will fit everywhere. Of course, you can still get those numeric values for cooperation with other CUDA-related software.
- Various [Plain Old Data](http://en.cppreference.com/w/cpp/concept/PODType) structs adorned with **convenience methods and operators** (e.g. device properties, block and grid dimensions).
- Aims for **clarity and straightforwardness** in naming and semantics, so that you don't need to refer to the official documentation to understand what each class and function do.
- Aims for conformance with the [C++ core guidelines](https://github.com/isocpp/CppCoreGuidelines).
- Thin and **lightweight**:
    - No work done behind your back, no caches or indices or any such thing - except in corner cases for ensuring Runtime-API and Driver-API compatibility. The sole exception is lazy creation of devices' primary context.
    - No costly inheritance structure, vtables, virtual methods and so on, for almost all wrappers; they vanishes almost entirely on compilation.
    - All "Runtime-API level" actions are implemented so as not to disrupt "Driver-API-level" work.
- **Header-only**: No need to compile anything.
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
* There is a very large number of API functions, many of are related, with similar names, and are easy to confuse.
* You will be working with a lot of numeric and pointer-based handles, instead of class or structs. Other than the poor aesthetics, this makes it easy to mix up resource-handle and pure-number parameters to functions.

You may have noticed this list reads like the opposite of the [key features](#key-features), listed above: The idea is to make this library overcome and rectify all of these deficiencies as much as possible.


## Detailed documentation

Detailed Doxygen-genereated documentation is [available](https://codedocs.xyz/eyalroz/cuda-api-wrappers/). It is mostly complete for the Runtime API wrappers, less so for the rest of the wrappers.

## Requirements

- CUDA: v11.x or later recommended, v9.0 or later supported.

    Remember that an NVIDIA driver compatible with your CUDA version also needs to be installed. Typically, this can be the one bundled in your CUDA distribution itself.

- Other software:
     - A C++11-capable compiler compatible with your version of CUDA.
     - CMake v3.18 or later; it's very easy to [download and install](https://cmake.org/download/) a recent version - no need to build it yourself.

- An NVIDIA GPU supporting Unified Virtual Addressing (UVA), i.e. Fermi microarchitecture or later. With earlier GPUs, memory copying, and other functionality relying on automtically determining where a memory address is located, will fail.

## Using the library in your project

**Use involving CMake:**

* Use CMake to configure, build and install the library. Then, in another CMake project, use `find_package(cuda_api_wrappers)` and make sure the library's install location is in CMake's package search path. This will let you use three targets within the `cuda-api-wrappers::` namespace: `runtime-and-driver`, `nvrtc` and `nvtx`.
* Use CMake's `FetchContent` module to obtain the project source code and make it part of your own project's build, e.g.:
   ```
   include(FetchContent)
   FetchContent_Declare(cuda-api-wrappers_library
       GIT_REPOSITORY https://github.com/eyalroz/cuda-api-wrappers.git
       GIT_TAG v12.34.56 # Replace this with a real available version
   )
   FetchContent_MakeAvailable(cuda-api-wrappers_library)
   ```
   The _same_ target names, with the namespaces, will be available in this case.

**Use not involving CMake:**

* Since this is a header-only library, you can simply add the `src/` subdirectory as one of your project's include directories. However, if you do this, it will be up to you to make sure and have the CUDA include directory in you include path as well, and to link against the CUDA driver, runtime API, nvrtc and/or nvtx libraries as appropriate.

Finally, if you've started using the library in a publicly-available (FOSS or commercial) project, please consider emailing [@eyalroz](https://github.com/eyalroz), or open an [issue](https://github.com/eyalroz/printf/issues/), to announce this.

## Coverage of the APIs

Most, but not all, API calls in the Runtime, Driver, NVTX and NVRTC are covered by these wrappers. Specifically, the following are missing:

* Execution graph management
* Interoperability with OpenGL, Direct3D, EGL, VDAPU.

Support for textures, arrays and surfaces exists, but is partial: Not all relevant API functions are covered.

The [Milestones](https://github.com/eyalroz/cuda-api-wrappers/milestones) indicates some features which aren't covered and are slated for future work. Since I am not currently working on anything graphics-related, there are no short-term plans to extend coverage to more graphics-related APIs; however - PRs are welcome.

## A taste of some features in play

We've all dreamed of being able to type in:

	auto callback =	[&foo] { std::cout << "Hello " << foo << " world!\n"; }
	my_stream.enqueue.host_invokable(callback);

... and have that just work, right? Well, now it does!

On a slightly more serious note, though, let's demonstrate the principles listed above:

#### Use of namespaces (and internal classes)
With this library, you would do `cuda::memory::host::allocate()` instead of `cudaMallocHost()` or `cuMemAllocHost()` and `cuda::device_t::memory::allocate()` instead of setting the current device and then `cudaMalloc()` or `cuMemAlloc()`. Note, though, that `device_t::memory::allocate()` is not a freestanding function but a method of an internal class, so a call to it might be `cuda::device::get(my_device_id).memory.allocate(my_size)`. The compiled version of this supposedly complicated construct will be nothing but the sequence of API calls: `cuInit()`, `cuDevicePrimaryCtxRetain()`, `cuCtxPushCurrent()`, `cuMemAlloc()` etc.

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

* If you're already familiar with the library, and want to help test new features and improvements, or help otherwise - please [email me](mailto:eyalroz1@gmx.com).
* If you notice a bug, compatibility problem, missing functionality or other problem - please [file the issue](https://github.com/eyalroz/cuda-api-wrappers/issues) here on GitHub. If you'd like to give less public feedback -  you can do that [via email](mailto:eyalroz1@gmx.com).
* You can also write if you're interested in collaborating on related research or coding work.
