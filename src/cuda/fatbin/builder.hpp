/**
 * @file
 *
 * @brief Contains the @ref fatbin_builder_t class and related code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_FATBIN_BUILDER_HPP_
#define CUDA_API_WRAPPERS_FATBIN_BUILDER_HPP_

#if CUDA_VERSION >= 12040

#include "../api/detail/region.hpp"
#include "builder_options.hpp"
#include "types.hpp"

#include <string>

namespace cuda {

///@cond
class fatbin_builder_t;
///@endcond

namespace fatbin_builder {

inline fatbin_builder_t wrap(handle_t handle, bool take_ownership = false) noexcept;

inline fatbin_builder_t create(const options_t & options);

namespace detail_ {

inline ::std::string identify(handle_t handle)
{
	return "Fatbin builder with handle " + cuda::detail_::ptr_as_hex(handle);
}

inline ::std::string identify(const fatbin_builder_t&);

} // namespace detail_

} // namespace fatbin_builder


class fatbin_builder_t {
public: // type definitions
	using size_type = ::size_t;

	struct deleter_type {
		void operator()(void * data) const { operator delete(data); }
	};

public: // getters

	fatbin_builder::handle_t handle() const
	{ return handle_; }

	/// True if this wrapper is responsible for telling CUDA to destroy
	/// the fatbin handle upon the wrapper's own destruction
	bool is_owning() const noexcept
	{ return owning; }

protected: // unsafe actions

	void build_without_size_check_in(memory::region_t target_region) const
	{
		auto status = nvFatbinGet(handle_, target_region.data());
		throw_if_error_lazy(status, "Failed completing the generation of a fatbin at " +
			cuda::detail_::ptr_as_hex(target_region.data()));
	}

public:
	size_type size() const
	{
		size_type result;
		auto status = nvFatbinSize(handle_, &result);
		throw_if_error_lazy(status, "Failed determining prospective fatbin size for " + fatbin_builder::detail_::identify(*this));
		return result;
	}

	void build_in(memory::region_t target_region) const
	{
		auto required_size = size();
		if (target_region.size() < required_size) {
			throw ::std::invalid_argument("Provided region for fatbin creation is of size "
				+ ::std::to_string(target_region.size()) + " bytes, while the fatbin requires " + ::std::to_string(required_size));
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
			throw ::std::invalid_argument("Empty PTX source code passed for addition into fatbin");
		}
		if (nul_terminated_ptx_source[nul_terminated_ptx_source.size() - 1] != '\0') {
			throw ::std::invalid_argument("PTX source code passed for addition into fatbin was not nul-character-terminated");
		}
#endif
		auto compute_capability_str = ::std::to_string(target_compute_capability.as_combined_number());
		auto empty_cmdline = "";
		auto status = nvFatbinAddPTX(handle_,
			nul_terminated_ptx_source.data(),
			nul_terminated_ptx_source.size(),
			compute_capability_str.c_str(),
			identifier,
			empty_cmdline);
		throw_if_error_lazy(status, "Failed adding PTX source fragment "
			+ ::std::string(identifier) + " at " + detail_::ptr_as_hex(nul_terminated_ptx_source.data())
			+ " to a fat binary for target compute capability " + compute_capability_str);
	}

	void add_lto_ir(
		const char* identifier,
		memory::region_t lto_ir,
		device::compute_capability_t target_compute_capability) const
	{
		auto compute_capability_str = ::std::to_string(target_compute_capability.as_combined_number());
		auto empty_cmdline = "";
		auto status = nvFatbinAddLTOIR(
			handle_, lto_ir.data(), lto_ir.size(), compute_capability_str.c_str(), identifier, empty_cmdline);
		throw_if_error_lazy(status, "Failed adding LTO IR fragment "
			+ ::std::string(identifier) + " at " + detail_::ptr_as_hex(lto_ir.data())
			+ " to a fat binary for target compute capability " + compute_capability_str);
	}

	void add_cubin(
		const char* identifier,
		memory::region_t cubin,
		device::compute_capability_t target_compute_capability) const
	{
		auto compute_capability_str = ::std::to_string(target_compute_capability.as_combined_number());
		auto status = nvFatbinAddCubin(
			handle_, cubin.data(), cubin.size(), compute_capability_str.c_str(), identifier);
		throw_if_error_lazy(status, "Failed adding cubin fragment "
			+ ::std::string(identifier) + " at " + detail_::ptr_as_hex(cubin.data())
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
		throw_if_error_lazy(status, "Failed adding relocatable PTX code at " + detail_::ptr_as_hex(ptx_code.data())
									+ "to fatbin builder " + fatbin_builder::detail_::identify(*this) );
	}

	// TODO: WTF is an index?
	void add_index(const char* identifier, memory::region_t index) const
	{
		auto status = nvFatbinAddIndex(handle_, index.data(), index.size(), identifier);
		throw_if_error_lazy(status, "Failed adding index  " + ::std::string(identifier) + " at "
			+ detail_::ptr_as_hex(index.data()) + " to a fat binary");
	}
#endif // CUDA_VERSION >= 12050

protected: // constructors

	fatbin_builder_t(
		fatbin_builder::handle_t handle,
		// no support for options, for now
		bool take_ownership) noexcept
		: handle_(handle), owning(take_ownership)
	{}

public: // friendship

	friend fatbin_builder_t fatbin_builder::wrap(fatbin_builder::handle_t, bool) noexcept;

public: // constructors and destructor

	fatbin_builder_t(const fatbin_builder_t &) = delete;

	fatbin_builder_t(fatbin_builder_t &&other) noexcept:
		fatbin_builder_t(other.handle_, other.owning)
	{
		other.owning = false;
	};

	~fatbin_builder_t() DESTRUCTOR_EXCEPTION_SPEC
	{
		if (not owning) { return; }

		auto status = nvFatbinDestroy(&handle_); // this nullifies the handle :-O
#ifdef THROW_IN_DESTRUCTORS
		throw_if_error_lazy(status,
			::std::string("Failed destroying fatbin builder ") + detail_::ptr_as_hex(handle_) +
			" in " + fatbin_builder::detail_::identify(handle_));
#else
		(void) status;
#endif

	}

public: // operators

	fatbin_builder_t &operator=(const fatbin_builder_t &) = delete;

	fatbin_builder_t &operator=(fatbin_builder_t &&other) noexcept
	{
		::std::swap(handle_, other.handle_);
		::std::swap(owning, owning);
		return *this;
	}

protected: // data members
	fatbin_builder::handle_t handle_;
	bool owning;
	// this field is mutable only for enabling move construction; other
	// than in that case it must not be altered
};

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

namespace detail_ {

inline ::std::string identify(const fatbin_builder_t& builder)
{
	return identify(builder.handle());
}

} // namespace detail_

} // namespace fatbin_builder


} // namespace cuda

#endif // CUDA_VERSION >= 12040

#endif // CUDA_API_WRAPPERS_FATBIN_BUILDER_HPP_
