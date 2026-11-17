/**
 * @file
 *
 * @brief An importation of an C++-17-like @ref cuda_::optional class and related definitions.
 *
 * @note When compiling with C++17 or later, the actual @ref std::optional class is used.
 */
#ifndef CAW_WRAPPERS_UTIL_OPTIONAL_HPP_
#define CAW_WRAPPERS_UTIL_OPTIONAL_HPP_

#if __cplusplus >= 201703L
#include <optional>
#include <any>
namespace cuda_ {
using std::optional;
using std::nullopt_t;
using std::nullopt;
using std::make_optional;
using std::in_place;
using std::in_place_t;
using std::bad_optional_access;
} // namespace cuda_
#else
#include "optional_lite.hpp"
namespace cuda_ {
using nonstd::optional;
using nonstd::nullopt_t;
using nonstd::nullopt;
using nonstd::make_optional;
using nonstd::in_place;
using nonstd::in_place_t;
using nonstd::bad_optional_access;
} // namespace cuda_
#endif // __cplusplus >= 201703L

#endif // CAW_WRAPPERS_UTIL_OPTIONAL_HPP_
