/**
 * @file
 *
 * @brief A smart pointer for CUDA device- and host-side memory, similar
 * to the standard library's <a href="http://en.cppreference.com/w/cpp/memory/unique_ptr">::std::unique_ptr</a>.
 *
 * @note Unique pointers, like any (wrapped) memory allocations, do _not_ extend the lifetime of
 * contexts (primary or otherwise). In particular, they do not increase primary context refcounts.
 *
 */
#ifndef CUDA_API_WRAPPERS_UNIQUE_REGION_HPP_
#define CUDA_API_WRAPPERS_UNIQUE_REGION_HPP_

#include "memory.hpp"
#include <cassert>

namespace cuda {
namespace memory {

/**
 * A class for holding a @ref region_t of memory owned "uniquely" by
 * its creator - similar to how `::std::unique_ptr` holds a uniquely-
 * owned pointer.
 *
 * @note The class is not templated on the element type - since that
 * is quite immaterial to its management (as well as its copying etc.)
 *
 * @tparam Deleter Similar to @ref ::std::unique_ptr's Deleter parameter;
 * it needs to be default-constructible and have an operator().
 *
 * @todo : Should we really expose the region parent class? We could,
 * instead, have a `get()` or `region()` method which provides lvalue-ref
 * access to it.
 *
 * @todo: Should this be called a `unique_region`, a-la-`unique_ptr`? Or
 * perhaps `unique_ptr` is a misnomer, and should have been called `owned_ptr`
 * (as opposed to `shared_ptr`), and regardless, this should be called an
 * `owned_region`?
 */
template<typename Deleter>
class unique_region : public region_t {
public: // types
    using parent = region_t;
    using region_t::pointer;
    using region_t::const_pointer;
    using region_t::size_type;
    using deleter_type = Deleter;
    // and _no_ element_type!

public:

    /// Default constructor, creates an empty unique_region which owns nothing
    constexpr unique_region() noexcept = default;

    /// Act like the default constructor for nullptr_t's
    constexpr unique_region(::std::nullptr_t) noexcept : unique_region() { }

    /// Take ownership of an existing region
    explicit unique_region(region_t region) noexcept : region_t{region} { }

    // Note: No constructor which also takes a deleter. We do not hold a deleter
    // member - unlike unique_ptr's. If we wanted a general-purpose unique region
    // that's not just GPU allcoation-oriented, we might have had one of those.

    /// Move constructor.
    unique_region(unique_region&& other) noexcept : unique_region(other.release()) { }
    // Disable copy construction
    unique_region(const unique_region&) = delete;

    // Note: No conversion from "another type" like with ::std::unique_pointer, since
    // this class is not variant with the element type; and there's not much sense in
    // supporting conversion of memory between different deleters (/ allocators).

    ~unique_region() noexcept
    {
        if (data() != nullptr) {
            deleter_type{}(data());
        }
		static_cast<region_t&>(*this) = region_t{ nullptr, 0 };
    }

    /// No copy-assignment - that would break our ownership guarantee
    unique_region& operator=(const unique_region&) = delete;

    /// A Move-assignment operator, which takes ownership of the other region
    unique_region& operator=(unique_region&& other) noexcept
    {
        reset(other.release());
        return *this;
    }

    // No "assignment from anoterh type", a s

    /// Reset the %unique_region to empty, invoking the deleter if necessary.
    unique_region&
    operator=(::std::nullptr_t) noexcept
    {
        reset();
        return *this;
    }

    /// No plain dereferencing - as there is no guarantee that any object has been
    /// initialized at those locations, nor do we know its type

    /// TODO: Should we support arrow-dereferencing?

	operator const_region_t() const noexcept { return *this; }

    /// Return the stored pointer.
    region_t get() const noexcept { return *this; }

    /// Return a deleter of the fixed type (it can't be a reference -
    /// we don't keep a deleter object)
    deleter_type get_deleter() const noexcept { return Deleter{}; }

    /// Return @c true if the stored pointer is not null.
    explicit operator bool() const noexcept { return data() != nullptr; }

    // Modifiers.

    /// Release ownership of any stored pointer.
    region_t release() noexcept
    {
        // TODO: Shouldn't I use move construction for release?
        region_t released { *this };
        static_cast<region_t&>(*this) = region_t{ nullptr, 0 };
        return released;
    }

    /** @brief Replace the memory region held by this object.
     *
     * @param region  The new region to maintain with unique ownership.
     *
     * @note The deleter is invoked on the previously-held region, if
     * one exists.
     */
    void reset(region_t region = region_t{})
    {
        ::std::swap<region_t>(*this, region);
        if (region.start() != nullptr) {
            get_deleter()(region);
        }
    }

    /// Exchange the pointer and deleter with another object.
    void swap(unique_region& other) noexcept
    {
        ::std::swap<region_t>(*this, other);
    }
}; // class unique_region

namespace device {

/// A unique region of device-global memory
using unique_region = memory::unique_region<detail_::deleter>;

namespace detail_ {

inline unique_region make_unique_region(const context::handle_t context_handle, size_t num_bytes)
{
    CAW_SET_SCOPE_CONTEXT(context_handle);
    return unique_region{ allocate_in_current_context(num_bytes) };
}

} // namespace detail_

/**
 * @brief Allocate a region in device-global memory
 *
 * @param context The context within which (and in the device global memory
 *     of which) to make the allocation
 * @param num_bytes Size of the region to be allocated, in bytes
 * @returns An owning RAII/CADRe object for the allocated memory region
 */
unique_region make_unique_region(const context_t& context, size_t num_bytes);

/**
 * @brief Allocate a region in device-global memory
 *
 * @param device The device in the global memory of which to make the allocation
 * @returns An owning RAII/CADRe object for the allocated memory region
 */
unique_region make_unique_region(const device_t& device, size_t num_bytes);

/**
 * @brief Allocate a region in device-global memory within the primary context
 * of the current CUDA device
 *
 * @param device The device in the global memory of which to make the allocation
 * @returns An owning RAII/CADRe object for the allocated memory region
 */
unique_region make_unique_region(size_t num_bytes);
///}@

} // namespace device


/// See @ref device::make_unique_region(const context_t& context, size_t num_elements)
inline device::unique_region make_unique_region(const context_t& context, size_t num_elements)
{
	return device::make_unique_region(context, num_elements);
}

/// See @ref device::make_unique_region(const device_t& device, size_t num_elements)
inline device::unique_region make_unique_region(const device_t& device, size_t num_elements)
{
	return device::make_unique_region(device, num_elements);
}

namespace host {

/// A unique region of pinned host memory
using unique_region = memory::unique_region<detail_::deleter>;

/**
 * @brief Allocate a physical-address-pinned region of system memory
 *
 * @returns An owning RAII/CADRe object for the allocated memory region
 */
inline unique_region make_unique_region(size_t num_bytes);

} // namespace host

namespace managed {

/// A unique region of managed memory, see @ref cuda::memory::managed
using unique_region = memory::unique_region<detail_::deleter>;

namespace detail_ {

inline unique_region make_unique_region(
    const context::handle_t  context_handle,
    size_t                   num_bytes,
    initial_visibility_t     initial_visibility = initial_visibility_t::to_all_devices)
{
    CAW_SET_SCOPE_CONTEXT(context_handle);
    return unique_region { allocate_in_current_context(num_bytes, initial_visibility) };
}

} // namespace detail_

/**
 * @copydoc make_unique_region(size_t num_bytes)
 *
 * @param context A context, to set when allocating the memory region, for whatever
 *     association effect that may have.
 */
inline unique_region make_unique_region(
    const context_t&      context,
    size_t                num_bytes,
    initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices);

/**
 * @copydoc make_unique_region(size_t num_bytes)
 *
 * @param device A context, whose primary context will be current when allocating
 *     the memory region, for whatever association effect that may have.
 */
inline unique_region make_unique_region(
    const device_t&       device,
    size_t                num_bytes,
    initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices);

/**
 * @brief Allocate a region of managed memory, accessible both from CUDA devices
 * and from the CPU.
 *
 * @returns An owning RAII/CADRe object for the allocated managed memory region
 */
inline unique_region make_unique_region(
    size_t                num_bytes);

} // namespace managed

} // namespace memory

} // namespace cuda

#endif // CUDA_API_WRAPPERS_UNIQUE_REGION_HPP_
