# cuda-api-wrappers:<br> Thin C++-flavored wrappers for the CUDA runtime API

Branch Build Status: Master [![Master Build Status](https://travis-ci.org/eyalroz/cuda-api-wrappers.svg?branch=master)](https://travis-ci.org/eyalroz/cuda-api-wrappers) | Development: [![Development Build Status](https://travis-ci.org/eyalroz/cuda-api-wrappers.svg?branch=development)](https://travis-ci.org/eyalroz/cuda-api-wrappers)

nVIDIA's [Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html) for [CUDA](http://www.nvidia.com/object/cuda_home_new.html) is intended for use both in C and C++ code. As such, it uses a C-style API, the lowest common denominator (with a few [notable exceptions](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html) of templated function overloads).

This library of wrappers around the Runtime API is intended to allow us to embrace many of the features of C++ (including some C++11) for using the runtime API - but without reducing expressivity or increasing the level of abstraction (as in, e.g., the [Thrust](https://thrust.github.io/) library). Using cuda-api-wrappers, you still have your devices, streams, events and so on - but they will be more convenient to work with in more C++-idiomatic ways.

## Key features

- All functions and methods throw **exceptions** on failure - no need to check return values (the exceptions carry the status information).
- Judicious **namespacing** (and some internal namespace-like classes) for better clarity and for semantically grouping related functionality together.
- There are **proxy/wrapper objects** for devices, streams, events, kernels and so on, using [RAII](http://en.cppreference.com/w/cpp/language/raii) to relieve you of remembering to free or destroy resources. 
- You can  **mostly forget about numeric IDs and handles**; the proxy classes will fit everywhere.
- Various [Plain Old Data](http://en.cppreference.com/w/cpp/concept/PODType) structs adorned with **convenience methods and operators**.
- Aims for **clarity and straightforwardness** in naming and semantics, so that you don't need to look concepts up in the official documentation to understand what each class and function do.
- Thin and **lightweight**: 
    - No work done behind your back, no caches or indices or any such thing.
    - No costly inheritance structure, vtables, virtual methods and so on - vanishes almost entirely on compilation.
    - Doesn't really "hide" any of CUDA's complexity or functionality; it only simplifies _use_ of the Runtime API.

## Detailed documentation

Detailed nearly-complete Doxygen-genereated documentation is [available](https://codedocs.xyz/eyalroz/cuda-api-wrappers/).

## Requirements

- CUDA v8.0 or later is recommended and v7.5 should be supported (but is untested). CUDA 6.x should probably be Ok as well.
- A C++11-capable compiler compatible with your version of CUDA.
- CMake v3.8 or later - although most of the library will work as simple headers with no building.

## Coverage of the Runtime API

Considering the [list of runtime API modules](http://docs.nvidia.com/cuda/cuda-runtime-api/modules.html#modules), the library currently has the following (w.r.t. CUDA 8.x):

| Coverage level  | Modules                                                                 | 
|-----------------|-------------------------------------------------------------------------| 
| full            | Error Handling, Stream Management, Event Management, Version Management, Peer Device Memory Access, Occupancy, Unified Addressing |
| almost full     | Device Management (no chooseDevice, cudaSetValidDevices),  Memory Management, Execution Control (no support for working with parameter buffers) |
| partial     | 2D & 3D Arrays, Texture Object Management, Texture Reference Management  |
| (deprecated)    | Thread management |
| no coverage     | Graph Management, OpenGL Interoperability, Direct3D Interoperability, VDPAU Interoperability, EGL Interoperability, Graphics Interoperability, Surface Reference Management, Surface Object Management   |

The [Milestones](https://github.com/eyalroz/cuda-api-wrappers/milestones) indicates some features which aren't covered and are slated for future work.

Since I am not currently working on anything graphics-related, there are no short-term plans to extend coverage to any of the graphics related modules.

## A taste of the key features in play

We've all dreamed of being able to type in:

	my_stream.enqueue.callback(
		[&foo](cuda::stream_t stream, cuda::status_t status) {
			std::cout << "Hello " << foo << " world!\n";
		}
	);

... and have that just work, right? Well, now it does! 

On a slightly more serious note, though, let's demonstrate the principles listed above:

#### Use of namespaces (and internal classes)
With this library, you would do `cuda::memory::host::allocate()` instead of `cudaMallocHost()` and `cuda::device_t::memory::allocate()` instead of setting the current device and then `cudaMalloc()`. Note, though, that `device_t::memory::allocate()` is not a freestanding function but a method of an internal class, so a call to it might be `cuda::device::get(my_device_id).memory.allocate(my_size)`. The compiled version of this supposedly complicated construct will be nothing but the sequence of `cudaSetDevice()` and `cudaMalloc()` calls.

#### Adorning POD structs with convenience methods
The expression 
```
my_device.properties().compute_capability() >= cuda::make_compute_capability(50)
```
is a valid comparison, true for all devices with a Maxwell-or-later micro-architecture. This, despite the fact that `struct cuda::compute_capability_t` is a POD type with two unsigned integer fields, not a scalar. Note that `struct cuda::device::properties_t` (which is really basically a `struct cudaDeviceProp` of the Runtime API itself) does not have a `compute_capability` field.

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

More detailed documentation / feature walk-through is forthcoming. For now I'm providing two kinds of short example programs; browsing their source you'll know essentially all there is to know about the API wrappers.

To build and run the examples (just as a sanity check), execute the following:

    [user@host:/path/to/cuda-api-wrappers/]$ cmake -S . -B build -DBUILD_EXAMPLES=ON . && cmake --build build/ && find build/examples/bin -exec "{}" ";"


#### Modified CUDA samples

The CUDA distribution contains sample programs demostrating various features and concepts. A few of these - which are not focused on device-side work - have been adapted to use the API wrappers - completely foregoing direct use of the CUDA Runtime API itself. You will find them in the [modified CUDA samples](https://github.com/eyalroz/cuda-api-wrappers/tree/master/examples/modified_cuda_samples/) example programs folder.

#### 'Coverage' test programs - by module of the Runtime API

Gradually, an example program is being added for each one of the CUDA Runtime API [Modules](http://docs.nvidia.com/cuda/cuda-runtime-api/modules.html#modules), in which the approach replacing use of those module API calls by use of the API wrappers is demonstrated. These per-module example programs can be found [here](https://github.com/eyalroz/cuda-api-wrappers/tree/master/examples/by_runtime_api_module/).

## Bugs, suggestions, feedback

I would like some help with building up documentation and perhaps a Wiki here; if you can spare the time - do [write me](mailto:eyalroz1@gmx.com). You can also do so if you're interested in collaborating on some related project or for general comments/feedback/suggestions.

If you notice a specific issue which needs addressing, especially any sort of bug or compilation error, please [file the issue](https://github.com/eyalroz/cuda-api-wrappers/issues) here on GitHub.
