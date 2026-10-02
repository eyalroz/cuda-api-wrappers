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

protected:
    bool has_token_;

    release_state_type release_state_;

public: // non-mutators
    operator bool() const noexcept { return has_token_; }
    bool has_token() const noexcept { return has_token_; }
    release_state_type release_state() const noexcept { return release_state_; }

public:// mutators
    void drop() noexcept { has_token_ = false; }
    release_state_type release() noexcept(noexcept(Release{}(release_state_)))
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
    ~token_holder() noexcept(noexcept(release())) { release(); }
    // refuse ownership on copy-assignment
    token_holder& operator=(token_holder const&) noexcept(noexcept(release()))
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
using handle_release = status_t (*)(Handle);

template <typename Handle>
struct handle_traits;

#define CAW_DEFINE_HANDLE_TRAITS(_handle_type, _contextualized, _release_func, _raw_release_func, _identify_func) \
template <> \
struct cuda_::detail::handle_traits<_handle_type> { \
    using handle_type = _handle_type; \
    using release_type = handle_release<handle_type>; \
    static constexpr bool contextualized = _contextualized; \
    static status_t release_nothrow(std::false_type, context::handle_t, handle_type handle) noexcept { \
        return _release_func(handle); \
    } \
    static status_t release_nothrow(std::true_type, context::handle_t context_handle, handle_type handle) noexcept { \
        CAW_SET_SCOPE_CONTEXT(context_handle); \
        return _release_func(handle); \
    } \
    static std::string identify(_handle_type handle) { return _identify_func(handle); } \
    static constexpr auto raw_release_func_name = CAW_STRINGIFY(_raw_release_func); \
};

template <typename Handle>
struct contextualized_handle_t { context::handle_t context_handle; Handle handle; };

// Q: Why is this templated on the wrapper type rather than the handle type?
// A: Because different wrappers may have the same-type handle with different release
//    functions. Example: a void pointer or a memory region.
template <typename Wrapper>
struct handle_release_helper {
    void operator()(contextualized_handle_t<typename Wrapper::handle_type> handle_in_context) CAW_DESTRUCTOR_EXCEPTION_SPEC
    {
        using handle_type = typename Wrapper::handle_type;
        using traits = handle_traits<handle_type>;
        auto context_handle = handle_in_context.context_handle;
        auto handle = handle_in_context.handle;
        auto status = traits::release_nothrow(bool_constant<traits::contextualized>{}, context_handle, handle);
#ifdef CAW_THROW_IN_DESTRUCTORS
        throw_if_error_lazy(status, std::string{traits::raw_release_func_name} + " failed for "
            + traits::identify(handle) + (std::is_same<handle_type, context::handle_t>::value ? "" : " in "
            + cuda_::detail::identify(context_handle)) );
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
