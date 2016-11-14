# cuda-api-wrappers:<br> Thin C++-flavored wrappers for the CUDA runtime API

nVIDIA's [Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html) for [CUDA](http://www.nvidia.com/object/cuda_home_new.html) is intended for use both in C and C++ code. As such, it uses a C-style API, the lower common denominator (with a few [notable exceptions](docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html) of templated function overloads).

This library of wrappers around the Runtime API is intended to allow us to embrace many of the features of C++ (including some C++11) for using the runtime API - but without reducing expressivity or increasing the level of abstraction (as in, e.g., the [Thrust](https://thrust.github.io/) library). Using cuda-api-wrappers, you still have your devices, streams, events and so on - but they will be more convenient to work with in more C++-idiomatic ways.

## Key features

- All functions and methods throw **exceptions** on failure - no need to check return values (the exceptions carry the status information).
- Judicious **namespacing** (and some internal namespace-like classes) for better clarity and for semantically grouping related functionality together.
- There are **proxy objects** for devices, streams, events and so on, using [RAII](http://en.cppreference.com/w/cpp/language/raii) to relieve you of remembering to free or destroy resources.
- Various [Plain Old Data](http://en.cppreference.com/w/cpp/concept/PODType) structs adorned with **convenience methods and operators**.
- Aims for **clarity and straightforwardness** in naming and semantics, so that you don't need to refer to the official documentation to understand what each class and function do.
- Thin and **lightweight**: 
    - No work done behind your back, no caches or indices or any such thing.
    - No costly inheritance structure, vtables, virtual methods and so on - vanishes almost entirely on compilation
    - Doesn't "hide" any of CUDA's complexity or functionality; it only simplifies _use_ of the Runtime API.

## Requirements

- CUDA v8.0 is recommended and v7.5 is supported (but not tested as frequently). CUDA 6.x should probably be ok as well.
- A C++11-capable compiler compatible with your version of CUDA.
- CMake version 2.8 or later - although you don't really need it, you can just copy the `src/` directory into your own project and just make sure to compile the non-header-only parts.

## Coverage of the Runtime API

Considering the [list of runtime API modules](http://docs.nvidia.com/cuda/cuda-runtime-api/modules.html#modules), the library currently has the following:

| Coverage level  | Modules                                                                 | 
|-----------------|-------------------------------------------------------------------------| 
| full            | Error Handling, Stream Management, Event Management, Memory Management, Version Management, Peer Device Memory Access |
| almost full     | Device Management (no chooseDevice),  Execution Control (no support for working with parameter buffers) |
| partial         | (none at the moment)
| no coverage     | Occupancy, Unified Addressing, OpenGL Interoperability, Direct3D 9 Interoperability, Direct3D 10 Interoperability, Direct3D 11 Interoperability, VDPAU Interoperability, EGL Interoperability, Graphics Interoperability, Texture Reference Management, Surface Reference Management, Texture Object Management, Surface Object Management   |

Since the (main) developer is not currently working on anything graphics-related, there are no short-term plans to extend coverage to any of the graphics related modules. Other modules may well become supported.

## Examples

#### Use of namespaces (and internal classes)
With this library, you would do `cuda::memory::host::allocate()` instead of `cudaMallocHost()` and `cuda::device_t::memory::allocate()` instead of setting the current device and then `cudaMalloc()`. Note, though, that `device_t::memory::allocate()` is not a freestanding function but a method of an internal class, so a call to it might be `cuda::device::get(my_device_id).memory.allocate(my_size)`. The compiled version of this supposedly complicated construct will be nothing but the sequence of `cudaSetDevice()` and `cudaMalloc()` calls.

#### Adorning POD structs with convenience methods
The expression 
```
my_device_properties.compute_capability() >= cuda::make_compute_capability(50)
```
is a valid comparison, true for all devices with a Maxwell-or-later micro-architecture. This, despite the fact that `struct cuda::compute_capability_t` is a POD type with two unsigned integer fields, not a scalar. Note that `struct cuda::device::properties_t` (which is really basically a `struct cudaDeviceProp` of the Runtime API itself) does not have a `compute_capability` field.

#### Meaningful naming
Instead of using 
```
cudaError_t cudaEventCreateWithFlags(
    cudaEvent_t* event, 
    unsigned int flags) 
```
which requires you remember what you need to specify as flags and how, you construct a `cuda::event_t` proxy objecty, using the constructor 
```
cuda::event_t::event_t(
    bool uses_blocking_sync,
    bool records_timing      = event::do_record_timing,
    bool interprocess        = event::not_interprocess)
```
The default values here are `enum : bool`'s, which you should also use when constructing an `event_t` with non-default parameters. There is also the no-arguments `event_t()` constructor which calls `cudaEventCreate` without flags.

#### Modified CUDA samples

More detailed documentation / feature walk-through is forthcoming. For now, have a look at the [modified CUDA samples](https://github.com/eyalroz/cuda-api-wrappers/tree/master/examples/modified_cuda_samples) example folder, which adapts some of the CUDA sample code to use the runtime API only via this wrappers library.

To build and run the samples, do:

    [user@host:/path/to/cuda-api-wrappers/]$ cmake . && make examples && examples/scripts/run-all-examples

## Bugs, suggestions, feedback

Do not hesitate to file issues for bugs/feature requests/design suggestions. If you're interested in collaboration or for general feedback, please contact me:

Eyal Rozenberg [\<E.Rozneberg@cwi.nl\>](mailto:Eyal Rozenberg <E.Rozneberg@cwi.nl>)
