/**
 * @file
 *
 * @brief the @ref handle_ownership_t class
 */
#ifndef CUDA_API_WRAPPERS_HANDLE_OWNERSHIP_HPP
#define CUDA_API_WRAPPERS_HANDLE_OWNERSHIP_HPP

#include "token_holder.hpp"
#include "../current_context.hpp"

namespace cuda_ {
namespace detail {

template <typename Handle>
struct contextualized_handle_t { context::handle_t context_handle; Handle handle; };

// Q: Why is this templated on the wrapper type rather than the handle type?
// A: Because different wrappers may have the same-type handle with different release
//    functions. Example: a void pointer or a memory region.
template <typename Wrapper>
struct handle_release_helper {
    using handle_type = typename Wrapper::handle_type;
    static status_t release_nothrow(std::false_type, context::handle_t, handle_type handle) noexcept {
        return Wrapper::release_handle(handle);
    }
    static status_t release_nothrow(std::true_type, context::handle_t context_handle, handle_type handle) noexcept {
        CAW_SET_SCOPE_CONTEXT(context_handle);
        return Wrapper::release_handle(handle);
    }
    void operator()(contextualized_handle_t<handle_type> handle_in_context) const CAW_DESTRUCTOR_EXCEPTION_SPEC
    {
        auto context_handle = handle_in_context.context_handle;
        auto handle = handle_in_context.handle;
        static constexpr bool contextualized = has_context_method<Wrapper>::value;
        auto status = release_nothrow(bool_constant<contextualized>{}, context_handle, handle);
#ifdef CAW_THROW_IN_DESTRUCTORS
        static constexpr bool handle_type_is_not_unique =
            std::is_same<handle_type, void*>::value or
            std::is_same<handle_type, void const*>::value or
            std::is_same<handle_type, memory::region_t>::value or
            std::is_same<handle_type, context::handle_t>::value;
        using unique_handle_type = typename std::conditional<handle_type_is_not_unique,
            tagged<Wrapper, handle_type>, handle_type>::type;
        unique_handle_type unique_handle { handle };
        throw_if_error_lazy(status, "Handle release failed for "
            + cuda_::detail::identify(unique_handle)
            + (std::is_same<handle_type, context::handle_t>::value ? "" : " in " + cuda_::detail::identify(context_handle)) );
#else
        (void) status;
#endif
    }
};

/**
 * @brief a subclass of the ownership token class, tailored
 * to the kind of ownership tokens we need for CUDA entity wrapper classes,
 * which are created around some handle obtained from the CUDA API, and which
 * requires release on destruction.
 */
template <typename Wrapper>
class handle_ownership_t : public token_holder<
    handle_release_helper<Wrapper>,
    contextualized_handle_t<typename Wrapper::handle_type>>
{
    using handle_type = typename Wrapper::handle_type;
    using contextualized_handle_type = contextualized_handle_t<handle_type>;
    using parent_type = token_holder<handle_release_helper<Wrapper>, contextualized_handle_type>;
    using parent_type::parent_type;
};

// Each wrapper classes which owns a handle needs to define these, for the handle_ownership_t
// member to compile and work
#define CAW_DEFINE_HANDLE_RELEASE_MEMBERS(_handle_release_func, _raw_release_func) \
	protected: \
	static constexpr auto release_handle = _handle_release_func; \
	static constexpr auto raw_handle_release_function_name = CAW_STRINGIFY(_raw_release_func); \
	template <typename Wrapper> friend struct cuda_::detail::handle_release_helper;

} // namespace detail
} // namespace cuda_

#endif //CUDA_API_WRAPPERS_HANDLE_OWNERSHIP_HPP
