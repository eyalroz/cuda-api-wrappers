/**
* @file
 *
 * @brief Definition of the @ref token_holder_t class, a subclass tailored
 * to the kind of ownership tokens we need for CUDA entity wrapper classes,
 * named @handle_token_holder_t , and finally, a macro facilitating the
 * definition of relevant traits for each of these classes, that are necessary
 * for them to have ownership tokens.
 */
#ifndef CUDA_API_WRAPPERS_TOKEN_HOLDER_HPP_
#define CUDA_API_WRAPPERS_TOKEN_HOLDER_HPP_

#include "../types.hpp"
#include "../identify.hpp"
#include "../error.hpp"

#ifndef CAW_STRINGIFY
#define CAW_STRINGIFY(_q) #_q
#endif

namespace cuda_ {
namespace detail {

// Note:: This class does _not_ create tokens, or adds refcounts,
// or transfer tokens when it's copied. So trying to copy it
// just results is in a tokenless, "empty" object.
template <typename Release, typename ReleaseState>
class token_holder {
    using release_type = Release;
    using release_state_type = ReleaseState;
    static constexpr bool release_may_throw = noexcept(Release{}(std::declval<ReleaseState>()));

protected:
    bool has_token_;

    release_state_type release_state_;

public: // non-mutators
    operator bool() const noexcept { return has_token_; }
    bool has_token() const noexcept { return has_token_; }
    release_state_type release_state() const noexcept { return release_state_; }

public:// mutators
    void drop() noexcept { has_token_ = false; }
    release_state_type release() noexcept(release_may_throw)
    {
        if (has_token_) { Release{}(release_state_); }
        return release_state_;
    }

public: // constructors & operators
    token_holder(bool owning, release_state_type release_state) noexcept
        : has_token_(owning), release_state_(std::move(release_state)) {}
    token_holder() noexcept : token_holder(false, {}) { };
    // take ownership on move
    token_holder(token_holder&& other) noexcept
        : has_token_(other.has_token_), release_state_(other.release_state_)
    {
        other.drop();
    }
    ~token_holder() noexcept(release_may_throw) { release(); }
    // refuse ownership on copy-assignment
    token_holder& operator=(token_holder const&) noexcept(release_may_throw)
    {
        token_holder empty{};
        swap(*this, empty);
        return *this;
    }
    token_holder& operator=(token_holder&& other) noexcept
    {
        swap(*this, other);
        return *this;
    }
    friend void swap(token_holder& a, token_holder& b) noexcept
    {
        std::swap(a.has_token_, b.has_token_);
        std::swap(a.release_state_, b.release_state_);
    }
};

template <typename Handle>
struct handle_traits;

// Note: This definition breaks older GCC compilers (e.g. GCC 6.5.0), as it qualifies the template specialization
// in a namespace. To retain compatibility with such compilers, drop the cuda_::detail:: prefix, and place
// invocations within the appropriate namespace
#define CAW_DEFINE_HANDLE_TRAITS(_wrapper_type, _release_func, _raw_release_func) \
template <> \
struct cuda_::detail::handle_traits<_wrapper_type> { \
    using handle_type = typename _wrapper_type::handle_type; \
    static status_t release_nothrow(std::false_type, context::handle_t, handle_type handle) noexcept { \
        return _release_func(handle); \
    } \
    static status_t release_nothrow(std::true_type, context::handle_t context_handle, handle_type handle) noexcept { \
        CAW_SET_SCOPE_CONTEXT(context_handle); \
        return _release_func(handle); \
    } \
    static constexpr auto raw_release_func_name = CAW_STRINGIFY(_raw_release_func); \
};

template <typename Handle>
struct contextualized_handle_t { context::handle_t context_handle; Handle handle; };

// Q: Why is this templated on the wrapper type rather than the handle type?
// A: Because different wrappers may have the same-type handle with different release
//    functions. Example: a void pointer or a memory region.
template <typename Wrapper>
struct handle_release_helper {
    void operator()(contextualized_handle_t<typename Wrapper::handle_type> handle_in_context) const CAW_DESTRUCTOR_EXCEPTION_SPEC
    {
        using traits = handle_traits<Wrapper>;
        auto context_handle = handle_in_context.context_handle;
        auto handle = handle_in_context.handle;
        static constexpr bool contextualized = has_context_method<Wrapper>::value;
        auto status = traits::release_nothrow(bool_constant<contextualized>{}, context_handle, handle);
#ifdef CAW_THROW_IN_DESTRUCTORS
        using handle_type = typename Wrapper::handle_type;
        static constexpr bool handle_type_is_not_unique =
            std::is_same<handle_type, void*>::value or
            std::is_same<handle_type, const void*>::value or
            std::is_same<handle_type, memory::region_t>::value or
            std::is_same<handle_type, context::handle_t>::value;
        using unique_handle_type = typename std::conditional<handle_type_is_not_unique,
            tagged<Wrapper, handle_type>, handle_type>::type;
        unique_handle_type unique_handle { handle };
        throw_if_error_lazy(status, std::string{traits::raw_release_func_name} + " failed for "
            + cuda_::detail::identify(unique_handle)
            + (std::is_same<handle_type, context::handle_t>::value ? "" : " in " + cuda_::detail::identify(context_handle)) );
#else
        (void) status;
#endif
    }
};

template <typename Wrapper>
class handle_ownership_t : public token_holder<
    handle_release_helper<Wrapper>,
    contextualized_handle_t<typename Wrapper::handle_type>>
{
    using handle_type = typename Wrapper::handle_type;
    using contextualized_handle_type = contextualized_handle_t<typename Wrapper::handle_type>;
    using parent_type = token_holder<handle_release_helper<Wrapper>, contextualized_handle_type>;
    using parent_type::parent_type;
};

enum : bool {
    is_not_contextual = false,
    isnt_contextual = is_not_contextual,
    is_contextual = true
};

} // namespace detail
} // namespace cuda_

#endif // CUDA_API_WRAPPERS_TOKEN_HOLDER_HPP_
