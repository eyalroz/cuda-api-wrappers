
#ifndef SRC_CUDA_API_DETAIL_MARSHALLED_OPTIONS_HPP_
#define SRC_CUDA_API_DETAIL_MARSHALLED_OPTIONS_HPP_

#include <cuda/nvrtc/types.hpp>

#include <cstdlib>
#include <vector>
#include <string>
#include <cstring>
#if __cplusplus >= 201703L
#include <charconv>
#endif

namespace cuda {

namespace rtc {

namespace detail_ {


class marshalled_options_size_computer_t {
public:
	using size_type = size_t;

protected:
	size_type buffer_pos_ { 0 };
	size_type num_options_ { 0 };

public:

	void append_to_current(string_view sv)
	{
		buffer_pos_ += sv.size();
	}

	void append_to_current(const char *str)
	{
		append_to_current(string_view{str, ::std::strlen(str)});
	}

	void append_to_current(unsigned long long v)
	{
		char v_buffer[::std::numeric_limits<unsigned long long>::digits10 + 2];
#if __cplusplus >= 201703L
		auto result = ::std::to_chars(v_buffer, v_buffer + sizeof(v_buffer), v);
		auto num_chars = result.ptr - v_buffer;
#else
		auto num_chars = snprintf(v_buffer, sizeof(unsigned long long), "%llu", v);
		// Q: Why can't we just print directly into the buffer?
		// A: There is no buffer, this is the size computer.

#endif
		buffer_pos_ += num_chars;
	}

	void append_to_current(long v)
	{
		char v_buffer[::std::numeric_limits<long>::digits10 + 2];
#if __cplusplus >= 201703L
		auto result = ::std::to_chars(v_buffer, v_buffer + sizeof(v_buffer), v);
		auto num_chars = result.ptr - v_buffer;
#else
		auto num_chars = snprintf(v_buffer, sizeof(long), "%ld", v);
		// Q: Why can't we just print directly into the buffer?
		// A: There is no buffer, this is the size computer.

#endif
		buffer_pos_ += num_chars;
	}

	void append_to_current(unsigned long v)
	{
		char v_buffer[::std::numeric_limits<unsigned long>::digits10 + 2];
#if __cplusplus >= 201703L
		auto result = ::std::to_chars(int_buffer, int_buffer + sizeof(int_buffer), v);
		auto num_chars = result.ptr - int_buffer;
#else
		auto num_chars = snprintf(v_buffer, sizeof(v_buffer), "%lu", v);
		// Q: Why can't we just print directly into the buffer?
		// A: There is no buffer, this is the size computer.

#endif
		buffer_pos_ += num_chars;
	}

	void append_to_current(int v)
	{
		char v_buffer[::std::numeric_limits<int>::digits10 + 2];
#if __cplusplus >= 201703L
		auto result = ::std::to_chars(v_buffer, v_buffer + sizeof(v_buffer), v);
		auto num_chars = result.ptr - v_buffer;
#else
		auto num_chars = snprintf(v_buffer, sizeof(int), "%d", v);
		// Q: Why can't we just print directly into the buffer?
		// A: There is no buffer, this is the size computer.

#endif
		buffer_pos_ += num_chars;
	}

	void append_to_current(unsigned v)
	{
		char v_buffer[::std::numeric_limits<unsigned>::digits10 + 2];
#if __cplusplus >= 201703L
		auto result = ::std::to_chars(int_buffer, int_buffer + sizeof(int_buffer), v);
		auto num_chars = result.ptr - int_buffer;
#else
		auto num_chars = snprintf(v_buffer, sizeof(v_buffer), "%u", v);
		// Q: Why can't we just print directly into the buffer?
		// A: There is no buffer, this is the size computer.

#endif
		buffer_pos_ += num_chars;
	}

	void append_to_current(char)
	{
		buffer_pos_++;
	}

	void finalize_current()
	{
		buffer_pos_++; // for a '\0'.
		num_options_++;
	}

	void push_back()
	{
		finalize_current();
	}

	template <typename T, typename... Ts>
	void push_back(T&& e, Ts&&... rest)
	{
		append_to_current(::std::forward<T>(e));
		push_back(::std::forward<Ts>(rest)...);
	}


public:
	size_t num_options() const { return num_options_; }
	size_t buffer_size() const { return buffer_pos_; }
};

} // namespace detail_

/**
 * This class is necessary for realizing everything we need from
 * the marshalled options: Easy access using an array of pointers,
 * for the C API - and RAIIness for convenience and safety when
 * using the wrapper classes.
 */
class marshalled_options_t {
public:
	using size_type = size_t;

	marshalled_options_t(size_type max_num_options, size_type buffer_size)
		: buffer_(buffer_size), option_ptrs_(max_num_options), current_option_(buffer_.data())
	{
		if (max_num_options > 0) { option_ptrs_[0] = buffer_.data(); }
	}
	marshalled_options_t(const marshalled_options_t&) = default;
	marshalled_options_t(marshalled_options_t&&) = default;

protected:
	dynarray<char> buffer_;
	dynarray<char*> option_ptrs_;
	char* current_option_;

	size_type buffer_pos_ { 0 };
	size_type num_options_ { 0 };

	char* current_pos() { return buffer_.data() + buffer_pos_; }

public:

	friend struct compilation_options_t;

	void append_to_current(string_view sv)
	{
#ifndef NDEBUG
		if (num_options_ >= option_ptrs_.size()) {
			::std::runtime_error("Attempt to prepare an option beyond the maximum number of options");
		}
		auto remaining = buffer_.size() - buffer_pos_; // including space for a last '\0'
		if (sv.size() + 1 > remaining) {
			throw ::std::logic_error("Not enough buffer space left for C-string argument: "
				+ ::std::to_string(sv.size() + 1) + " > " + ::std::to_string(remaining));
		}
#endif
		::std::copy_n(sv.cbegin(), sv.size(), current_pos());
		buffer_pos_ += sv.size();
	}

	void append_to_current(const char *str)
	{
		append_to_current(string_view{str, ::std::strlen(str)});
	}


	void append_to_current(unsigned long long v)
	{
#ifndef NDEBUG
		if (num_options_ >= option_ptrs_.size()) {
			::std::runtime_error("Attempt to prepare an option beyond the maximum number of options");
		}
#endif
		constexpr const ::std::size_t v_size = ::std::numeric_limits<unsigned long long>::digits10;
#if __cplusplus >= 201703L
		auto result = ::std::to_chars(current_pos(), current_pos() + v_size, v);
		auto num_chars = result.ptr - current_pos();
#else
		auto num_chars = snprintf(current_pos(), v_size, "%llu", v);
#endif
		buffer_pos_ += num_chars;
	}

	void append_to_current(long v)
	{
#ifndef NDEBUG
		if (num_options_ >= option_ptrs_.size()) {
			::std::runtime_error("Attempt to prepare an option beyond the maximum number of options");
		}
#endif
		constexpr const ::std::size_t v_size = ::std::numeric_limits<long>::digits10;
#if __cplusplus >= 201703L
		auto result = ::std::to_chars(current_pos(), current_pos() + v_size, v);
		auto num_chars = result.ptr - current_pos();
#else
		auto num_chars = snprintf(current_pos(), v_size, "%ld", v);
#endif
		buffer_pos_ += num_chars;
	}

	void append_to_current(unsigned long v)
	{
#ifndef NDEBUG
		if (num_options_ >= option_ptrs_.size()) {
			::std::runtime_error("Attempt to prepare an option beyond the maximum number of options");
		}
#endif
		constexpr const ::std::size_t v_size = ::std::numeric_limits<unsigned long>::digits10;
#if __cplusplus >= 201703L
		auto result = ::std::to_chars(current_pos(), current_pos() + v_size, v);
		auto num_chars = result.ptr - current_pos();
#else
		auto num_chars = snprintf(current_pos(), v_size, "%lu", v);
#endif
		buffer_pos_ += num_chars;
	}

	void append_to_current(int v)
	{
#ifndef NDEBUG
		if (num_options_ >= option_ptrs_.size()) {
			::std::runtime_error("Attempt to prepare an option beyond the maximum number of options");
		}
#endif
		constexpr const ::std::size_t v_size = ::std::numeric_limits<int>::digits10;
#if __cplusplus >= 201703L
		auto result = ::std::to_chars(current_pos(), current_pos() + v_size, v);
		auto num_chars = result.ptr - current_pos();
#else
		auto num_chars = snprintf(current_pos(), v_size, "%d", v);
#endif
		buffer_pos_ += num_chars;
	}

	void append_to_current(unsigned v)
	{
#ifndef NDEBUG
		if (num_options_ >= option_ptrs_.size()) {
			::std::runtime_error("Attempt to prepare an option beyond the maximum number of options");
		}
#endif
		constexpr const ::std::size_t v_size = ::std::numeric_limits<unsigned>::digits10;
#if __cplusplus >= 201703L
		auto result = ::std::to_chars(current_pos(), current_pos() + v_size, v);
		auto num_chars = result.ptr - current_pos();
#else
		auto num_chars = snprintf(current_pos(), v_size, "%u", v);
#endif
		buffer_pos_ += num_chars;
	}

	void append_to_current(char ch)
	{
#ifndef NDEBUG
		if (num_options_ >= option_ptrs_.size()) {
			::std::runtime_error("Attempt to prepare an option beyond the maximum number of options");
		}
#endif
		buffer_[buffer_pos_++] = ch;
	}

	void finalize_current()
	{
#ifndef NDEBUG
		if (num_options_ >= option_ptrs_.size()) {
			::std::runtime_error("Attempt to insert options beyond the maximum supported by this structure (and probably - a duplicate option)");
		}
#endif
		buffer_[buffer_pos_++] = '\0';
		option_ptrs_[num_options_++] = current_option_;
		current_option_ = current_pos();
	}

	void push_back()
	{
		finalize_current();
	}

	template <typename T, typename... Ts>
	void push_back(T&& e, Ts&&... rest)
	{
		append_to_current(::std::forward<T>(e));
		push_back(::std::forward<Ts>(rest)...);
	}

public:
	span<const char*> option_ptrs() const {
		return span<const char*>{const_cast<const char**>(option_ptrs_.data()), num_options_};
	}
};

} // namespace rtc

} // namespace cuda


#endif /* SRC_CUDA_API_DETAIL_MARSHALLED_OPTIONS_HPP_ */

