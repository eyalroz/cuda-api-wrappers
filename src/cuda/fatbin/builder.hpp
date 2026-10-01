/**
 * @file
 *
 * @brief Contains the @ref fatbin_builder_t class and related code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_FATBIN_BUILDER_HPP_
#define CUDA_API_WRAPPERS_FATBIN_BUILDER_HPP_

#if CUDA_VERSION >= 12040

#include "../api/detail/token_holder.hpp"
#include "../api/detail/region.hpp"
#include "builder_options.hpp"
#include "types.hpp"

#include <string>

namespace cuda_ {

///@cond
class fatbin_builder_t;
///@endcond

namespace fatbin_builder {

inline fatbin_builder_t wrap(handle_t handle, bool take_ownership = false) noexcept;

inline fatbin_builder_t create(const options_t & options);

namespace detail {

inline std::string identify(handle_t handle)
{
	return "Fatbin builder with handle " + cuda_::detail::ptr_as_hex(handle);
}

inline std::string identify(const fatbin_builder_t&);

inline cuda_::status_t destroy_nothrow(handle_t handle) noexcept
{
	auto fb_status = nvFatbinDestroy(&handle);
	// TODO: Arrange it so that this can return its own status type
	auto named =
		((fb_status == status::success) ?
		cuda_::status::success : cuda_::status::unknown);
	return static_cast<cuda_::status_t>(named);
}

} // namespace detail

} // namespace fatbin_builder

class fatbin_builder_t {
public: // type definitions
	using handle_type = fatbin_builder::handle_t;
	using size_type = ::size_t;

	struct deleter_type {
		void operator()(void * data) const { operator delete(data); }
	};

public: // getters

	fatbin_builder::handle_t handle() const { return handle_; }

	/// True if this wrapper is responsible for telling CUDA to destroy
	/// the fatbin handle upon the wrapper's own destruction
	bool is_owning() const noexcept { return ownership_.has_token(); }

protected: // unsafe actions

	void build_without_size_check_in(memory::region_t target_region) const
	{
		auto status = nvFatbinGet(handle_, target_region.data());
		throw_if_error_lazy(status, "Failed completing the generation of a fatbin at " +
			cuda_::detail::ptr_as_hex(target_region.data()));
	}

public:
	size_type size() const
	{
		size_type result;
		auto status = nvFatbinSize(handle_, &result);
		throw_if_error_lazy(status, "Failed determining prospective fatbin size for " + fatbin_builder::detail::identify(*this));
		return result;
	}

	void build_in(memory::region_t target_region) const
	{
		auto required_size = size();
		if (target_region.size() < required_size) {
			throw std::invalid_argument("Provided region for fatbin creation is of size "
				+ std::to_string(target_region.size()) + " bytes, while the fatbin requires " + std::to_string(required_size));
		}
		return build_without_size_check_in(target_region);
	}

	memory::unique_region<deleter_type> build() const
	{
		auto size_ = size();
		auto ptr = operator new(size_);
		memory::region_t target_region{ptr, size_};
		build_in(target_region);
		return memory::unique_region<deleter_type>(target_region);
	}

	void add_ptx_source(
		const char* identifier,
		span<char> nul_terminated_ptx_source,
		device::compute_capability_t target_compute_capability) const  // no support for options, for now
	{
#ifndef NDEBUG
		if (nul_terminated_ptx_source.empty()) {
			throw std::invalid_argument("Empty PTX source code passed for addition into fatbin");
		}
		if (nul_terminated_ptx_source[nul_terminated_ptx_source.size() - 1] != '\0') {
			throw std::invalid_argument("PTX source code passed for addition into fatbin was not nul-character-terminated");
		}
#endif
		auto compute_capability_str = std::to_string(target_compute_capability.as_combined_number());
		auto empty_cmdline = "";
		auto status = nvFatbinAddPTX(handle_,
			nul_terminated_ptx_source.data(),
			nul_terminated_ptx_source.size(),
			compute_capability_str.c_str(),
			identifier,
			empty_cmdline);
		throw_if_error_lazy(status, "Failed adding PTX source fragment "
			+ std::string(identifier) + " at " + detail::ptr_as_hex(nul_terminated_ptx_source.data())
			+ " to a fat binary for target compute capability " + compute_capability_str);
	}

	void add_lto_ir(
		const char* identifier,
		memory::region_t lto_ir,
		device::compute_capability_t target_compute_capability) const
	{
		auto compute_capability_str = std::to_string(target_compute_capability.as_combined_number());
		auto empty_cmdline = "";
		auto status = nvFatbinAddLTOIR(
			handle_, lto_ir.data(), lto_ir.size(), compute_capability_str.c_str(), identifier, empty_cmdline);
		throw_if_error_lazy(status, "Failed adding LTO IR fragment "
			+ std::string(identifier) + " at " + detail::ptr_as_hex(lto_ir.data())
			+ " to a fat binary for target compute capability " + compute_capability_str);
	}

	void add_cubin(
		const char* identifier,
		memory::region_t cubin,
		device::compute_capability_t target_compute_capability) const
	{
		auto compute_capability_str = std::to_string(target_compute_capability.as_combined_number());
		auto status = nvFatbinAddCubin(
			handle_, cubin.data(), cubin.size(), compute_capability_str.c_str(), identifier);
		throw_if_error_lazy(status, "Failed adding cubin fragment "
			+ std::string(identifier) + " at " + detail::ptr_as_hex(cubin.data())
			+ " to a fat binary for target compute capability " + compute_capability_str);
	}

#if CUDA_VERSION >= 12050
	/**
	 * Adds relocatable PTX entries from a host object to the fat binary being built
	 *
	 * @param ptx_code PTX "host object". TODO: Is this PTX code in text mode? Something else?
	 *
	 * @note The builder's options (specified on creation) are ignored for these operations.
	 */
	void add_relocatable_ptx(memory::region_t ptx_code) const
	{
		auto status = nvFatbinAddReloc(handle_, ptx_code.data(), ptx_code.size());
		throw_if_error_lazy(status, "Failed adding relocatable PTX code at " + detail::ptr_as_hex(ptx_code.data())
									+ "to fatbin builder " + fatbin_builder::detail::identify(*this) );
	}

	// TODO: WTF is an index?
	void add_index(const char* identifier, memory::region_t index) const
	{
		auto status = nvFatbinAddIndex(handle_, index.data(), index.size(), identifier);
		throw_if_error_lazy(status, "Failed adding index  " + std::string(identifier) + " at "
			+ detail::ptr_as_hex(index.data()) + " to a fat binary");
	}
#endif // CUDA_VERSION >= 12050

protected: // constructors

	fatbin_builder_t(
		fatbin_builder::handle_t handle,
		// no support for options, for now
		bool take_ownership) noexcept
		: handle_(handle), ownership_({take_ownership, { context::detail::none, handle} })
	{}

public: // friendship

	friend fatbin_builder_t fatbin_builder::wrap(fatbin_builder::handle_t, bool) noexcept;

public: // constructors and operators
	fatbin_builder_t(const fatbin_builder_t &) = delete;
	fatbin_builder_t(fatbin_builder_t &&other) noexcept = default;
	fatbin_builder_t &operator=(const fatbin_builder_t &) = delete;
	fatbin_builder_t &operator=(fatbin_builder_t &&other) noexcept = default;

protected: // data members
	fatbin_builder::handle_t handle_;
	detail::handle_ownership_t<fatbin_builder_t> ownership_;
	// this field is mutable only for enabling move construction; other
	// than in that case it must not be altered
}; // class fatbin_builder_t

CAW_DEFINE_HANDLE_TRAITS(fatbin_builder_t::handle_type, isnt_contextual, fatbin_builder::detail::destroy_nothrow,
	nvFatbinDestroy, fatbin_builder::detail::identify);

namespace fatbin_builder {

/// Create a new link-process (before adding any compiled images or or image-files)
inline fatbin_builder_t create(const options_t & options)
{
	handle_t new_handle;
	auto marshalled_options = marshalling::marshal(options);
	auto option_ptrs = marshalled_options.option_ptrs();
	auto status = nvFatbinCreate(&new_handle, option_ptrs.data(), option_ptrs.size());
	throw_if_error_lazy(status, "Failed creating a new fatbin builder");
	auto do_take_ownership = true;
	return wrap(new_handle, do_take_ownership);
}

inline fatbin_builder_t wrap(handle_t handle, bool take_ownership) noexcept
{
	return fatbin_builder_t{handle, take_ownership};
}

namespace detail {

inline std::string identify(const fatbin_builder_t& builder)
{
	return identify(builder.handle());
}

} // namespace detail

} // namespace fatbin_builder


} // namespace cuda_

#endif // CUDA_VERSION >= 12040

#endif // CUDA_API_WRAPPERS_FATBIN_BUILDER_HPP_
