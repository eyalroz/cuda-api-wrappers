/**
* @file
 *
 * @brief Definition of the @ref token_holder_t class, encapsulating
 * the logic of possibly holding some token and behaving appropriately
 * on construction, copy, move and destruction.
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

} // namespace detail
} // namespace cuda_

#endif // CUDA_API_WRAPPERS_TOKEN_HOLDER_HPP_
