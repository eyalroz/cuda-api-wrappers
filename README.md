# cuda-api-wrappers:<br> C++-flavored wrappers for the CUDA runtime API

nVIDIA's [Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html) for CUDA is intended for use both in C and C++ code. As such, it uses a C-style API, the lower common denominator (with a few [notable exceptions](docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html) of templated function overloads).

These wrappers are intended to allow us to embrace many of the features of C++ (including some C++11) for using the runtime API - but without reducing expressivity or increasing the level of abstraction (as in, e.g., the [Thrust](https://thrust.github.io/) library). Using cuda-api-wrappers, you would still have your devices, streams, events, device properties etc. - but they will be more convenient to work with and better-compatible with other 'heavily C++-ish' code.

### Key overall features

- All functions and methods throw **exceptions** on failure - no need to check return values (the exceptions carry the status information).
- Judicious **namespacing** (and some internal namespace-like classes) for better clarity and for semantically grouping related functionality together.
- There are **proxy objects** for devices, streams, events and so on, using [RAII](http://en.cppreference.com/w/cpp/language/raii) to relieve you of remembering to free or destroy resources.
- Various [Plain Old Data](http://en.cppreference.com/w/cpp/concept/PODType) structs adorned with **convenience methods and operators**.
- Avoidance of obscure parameters which require you to go read the official documentation to get right; everything is as clear and understandable as I could make it.
    
### Examples

- Use of namespaces (and internal classes): `cuda::memory::host::allocate(my_size)` instead of `cudaMallocHost()` and `cuda::device::get(my_device_id).memory.allocate(my_size)` instead of setting the current device and then `cudaMalloc()`.
- Adorning POD structs with convenience methods: The expression `my_device_properties.compute_capability() >= cuda::make_compute_capability(50)` is a valid comparison, true for all devices with a Maxwell-or-later micro-architecture. The expression is valid despite the fact that `struct compute_capability_t` is a POD type with 2 unsigned integer fields, and that `cudaDeviceProp` doesn't have the compute capability as a single field.
- Meaningful naming: Instead of using `cudaError_t cudaEventCreateWithFlags (* my_event_id, my_flags )`, which requires you remember what you need to specify as flags and how, you construct an `event_t` proxy objecty, using the constructor ```event_t(
		bool uses_blocking_sync, bool records_timing = do_record_timing, bool interprocess = not_interprocess)```. The default values are `enum : bool`'s, which are clearer to use also in your code as opposed to `event_t(true, false, false)`. There is also the no-arguments `event_t()` constructor which calls `cudaEventCreate` without flags.

More detailed documentation / feature walk-through is forthcoming. For now, have a look at the [Modified CUDA samples](https://github.com/eyalroz/cuda-api-wrappers/tree/master/examples/modified_cuda_samples) example folder, which adapts some of the CUDA sample code to use the runtime API only via this wrappers library.

### Bugs, suggestions, feedback

Do not hesitate to file issues for bugs/feature requests/design suggestions. If you're interested in collaboration or for general feedback, please contact me:

Eyal Rozenberg [\<E.Rozneberg@cwi.nl\>](mailto:Eyal Rozenberg <E.Rozneberg@cwi.nl>)
