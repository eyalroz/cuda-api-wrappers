/**
 * @file
 *
 * @brief Wrappers for linking modules of compiled CUDA code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_LINK_HPP_
#define CUDA_API_WRAPPERS_LINK_HPP_

#include "current_context.hpp"
#include "link_options.hpp"
#include "memory.hpp"
#include "module.hpp"
#include <cuda.h>

#if __cplusplus >= 201703L
#include <filesystem>
#endif

namespace cuda {

///@cond
class device_t;

class module_t;

class link_t;
///@endcond

namespace link {

enum class input_kind_t {
	cubin,    /// Compiled device-class-specific device code
	ptx,      /// PTX (microarchitecture-inspecific intermediate representation)
	fatbin,   /// A bundle of multiple cubin and/or PTX inputs; typically
	object,   /// A host-side binary object with embedded device code; a `.o` file
	library,  /// An archive of objects files with embedded device code; a `.a` file
};

using handle_t = CUlinkState;

// TODO: Check if the linking has been completed!
inline link_t wrap(
	device::id_t device_id,
	context::handle_t context_handle,
	link::handle_t handle,
	const link::options_t &options,
	bool take_ownership = false) noexcept;

inline link_t create(const void *image, const link::options_t &options);

// TODO: Use a clase-class with C++17 of later, made up of the two classes here
namespace input {

/**
 * A typed, named, image in memory which can be used as an input to a runtime
 * CUDA linking process.
 */
struct image_t : memory::region_t {
	const char *name;
	link::input_kind_t type;
};

struct file_t {
	const char *path; // TODO: Use a proper path in C++14 and later
	link::input_kind_t type;
};

} // namespace input

} // namespace link

/**
 * @brief Wrapper class for a CUDA link (a process of linking compiled code together into an
 * executable binary, using CUDA, at run-time)
 *
 * @note This class is a "reference type", not a "value type". Therefore, making changes
 * to the link is a const-respecting operation on this class.
 */
class link_t {

public: // getters
	/// The raw CUDA ID for the device w.r.t. which the event is defined
	device::id_t device_id() const noexcept
	{ return device_id_; };

	/// The raw CUDA handle for the context in which the represented stream is defined.
	context::handle_t context_handle() const noexcept
	{ return context_handle_; }

	/// True if this wrapper is responsible for telling CUDA to destroy the event upon the wrapper's own destruction
	bool is_owning() const noexcept
	{ return owning; }

	/// The device w.r.t. which the event is defined
	device_t device() const;

	/// The context in which this stream was defined.
	context_t context() const;

public:
	/**
	 * Complete a linking process, producing a completely-linked cubin image (for loading into
	 * modules).
	 *
	 * @return The completely-linked cubin image, in a sized memory range. This memory is owned
	 * by the link object, and must not be freed/deleted.
	 */
	memory::region_t complete() const
	{
		void *cubin_output_start;
		size_t cubin_output_size;
		auto status = cuLinkComplete(handle_, &cubin_output_start, &cubin_output_size);
		throw_if_error_lazy(status,
			"Failed completing the link with state at address " + cuda::detail_::ptr_as_hex(handle_));
		return memory::region_t{cubin_output_start, cubin_output_size};
	}

	// TODO: Replace this with methods which take wrapper classes.
	void add(link::input::image_t image, const link::options_t &ptx_compilation_options = {}) const
	{
		auto marshalled_options = marshal(ptx_compilation_options);
		auto status = cuLinkAddData(
			handle_,
			static_cast<CUjitInputType>(image.type),
			image.data(), // TODO: Is this really safe?
			image.size(),
			image.name,
			marshalled_options.count(),
			const_cast<link::option_t *>(marshalled_options.options()),
			const_cast<void **>(marshalled_options.values())
		);
		throw_if_error_lazy(status,
			"Failed adding input " + ::std::string(image.name) + " of type "
			+ ::std::to_string((int) image.type) + " to a link.");
	}

	void add_file(link::input::file_t file_input, const link::options_t &options) const
	{
		auto marshalled_options = marshal(options);
		auto status = cuLinkAddFile(
			handle_,
			static_cast<CUjitInputType_enum>(file_input.type),
			file_input.path,
			marshalled_options.count(),
			const_cast<link::option_t *>(marshalled_options.options()),
			const_cast<void **>(marshalled_options.values())
		);
		throw_if_error_lazy(status,
			"Failed loading an object of type " + ::std::to_string((int) file_input.type)
			+ " from file " + file_input.path);
	}

#if __cplusplus >= 201703L
	void add_file(const ::std::filesystem::path& path, link::input_kind_t file_contents_type) const
	{
		return add_file(path.c_str(), file_contents_type);
	}
#endif

protected: // constructors

	link_t(
		device::id_t device_id,
		context::handle_t context,
		link::handle_t handle,
		const link::options_t &options,
		bool take_ownership) noexcept
		: device_id_(device_id), context_handle_(context), handle_(handle), options_(options), owning(take_ownership)
	{}

public: // friendship

	friend link_t link::wrap(device::id_t, context::handle_t, link::handle_t, const link::options_t &, bool) noexcept;

public: // constructors and destructor

	link_t(const link_t &) = delete;

	link_t(link_t &&other) noexcept:
		link_t(other.device_id_, other.context_handle_, other.handle_, other.options_, other.owning)
	{
		other.owning = false;
	};

	~link_t() noexcept(false)
	{
		if (owning) {
			context::current::detail_::scoped_override_t set_context_for_this_scope(context_handle_);
			auto status = cuLinkDestroy(handle_);
			throw_if_error_lazy(status,
				::std::string("Failed destroying the link ") + detail_::ptr_as_hex(handle_) +
				" in " + context::detail_::identify(context_handle_, device_id_));
		}
	}

public: // operators

	link_t &operator=(const link_t &) = delete;

	link_t &operator=(link_t &&other) noexcept
	{
		::std::swap(device_id_, other.device_id_);
		::std::swap(context_handle_, other.context_handle_);
		::std::swap(handle_, other.handle_);
		::std::swap(options_, other.options_);
		::std::swap(owning, owning);
		return *this;
	}

protected: // data members
	device::id_t device_id_;
	context::handle_t context_handle_;
	link::handle_t handle_;
	link::options_t options_;
	bool owning;
	// this field is mutable only for enabling move construction; other
	// than in that case it must not be altered
};

namespace link {

inline link_t create(const link::options_t &options = link::options_t{})
{
	handle_t new_link_handle;
	auto marshalled_options = marshal(options);
	auto status = cuLinkCreate(
		marshalled_options.count(),
		const_cast<link::option_t *>(marshalled_options.options()),
		const_cast<void **>(marshalled_options.values()),
		&new_link_handle
	);
	throw_if_error_lazy(status, "Failed creating a new link ");
	auto do_take_ownership = true;
	auto context_handle = context::current::detail_::get_handle();
	auto device_id = context::current::detail_::get_device_id();
	return wrap(
		device_id,
		context_handle,
		new_link_handle,
		options,
		do_take_ownership);
}

// TODO: Check if the linking has been completed!
inline link_t wrap(
	device::id_t device_id,
	context::handle_t context_handle,
	link::handle_t handle,
	const link::options_t &options,
	bool take_ownership) noexcept
{
	return link_t{device_id, context_handle, handle, options, take_ownership};
}

} // namespace link

} // namespace cuda

#endif // CUDA_API_WRAPPERS_LINK_HPP_
