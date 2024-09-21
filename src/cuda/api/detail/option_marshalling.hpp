#ifndef SRC_CUDA_API_DETAIL_OPTION_MARSHALLING_HPP_
#define SRC_CUDA_API_DETAIL_OPTION_MARSHALLING_HPP_

#include "../types.hpp"

#include <cstdlib>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>
#include <algorithm>

namespace cuda {

namespace marshalling {

namespace detail_ {

// These two structs are streamed to a marshalled options object (see below)
// to indicate the start of a new option or the conclusion of all options,
// respectively

template <typename Delimiter> struct opt_start_t;

template<typename T>
struct is_marshalling_control : ::std::false_type {};

/**
 * This class is necessary for realizing everything we need from
 * the marshalled options: Easy access using an array of pointers,
 * for the C API - and RAIIness for convenience and safety when
 * using the wrapper classes.
 */
class marshalled_options_t {
public:
	using size_type = size_t;

	struct advance_gadget {}; /// triggers an advance() when streamed into an object of this class

	marshalled_options_t()
	{
		option_positions.push_back(0);
	}
	marshalled_options_t(size_type max_num_options)
	{
		option_positions.reserve(max_num_options);
		option_positions.push_back(0);
	}

	marshalled_options_t(const marshalled_options_t&) = delete;
	marshalled_options_t(marshalled_options_t&&) = default;

protected:
	mutable ::std::ostringstream oss;
	mutable ::std::string finalized {};
	::std::vector<size_type> option_positions;
	// Offsets into the eventually-created options string.
	// Note that the last offset is to a not-yet-existing option
public:

	bool empty() const {
		return oss.tellp() == 0;
	}

	template <typename T, typename = ::cuda::detail_::enable_if_t<not detail_::is_marshalling_control<typename ::std::decay<T>::type>::value>>
	marshalled_options_t& operator<<(T&& x)
	{
		oss << x;
		return *this;
	}

	marshalled_options_t& advance()
	{
		oss << '\0';
		option_positions.push_back(oss.tellp());
		return *this;
	}

	::std::vector<const char*> option_ptrs() const {
		finalized = oss.str();
		auto ptrs = ::std::vector<const char*>();
		ptrs.reserve(option_positions.size()-1);
		const char* start = finalized.data();
		::std::transform(option_positions.cbegin(), option_positions.cend() - 1, ::std::back_inserter(ptrs),
			[start] (size_type pos){ return start + pos; });
		return ptrs;
	}
};

inline marshalled_options_t& operator<< (marshalled_options_t& mo, marshalled_options_t::advance_gadget)
{
	mo.advance();
	return mo;
}

template<> struct is_marshalling_control<marshalled_options_t::advance_gadget> : ::std::true_type {};

template<typename Delimiter>
struct is_marshalling_control<opt_start_t<Delimiter>> : ::std::true_type {};

template <typename Delimiter>
struct opt_start_t {
	bool      ever_used;
	Delimiter delimiter_;

	opt_start_t(Delimiter delimiter) : ever_used(false), delimiter_(delimiter){ }
};

template <typename MarshalTarget, typename Delimiter>
MarshalTarget& operator<<(MarshalTarget& mt, detail_::opt_start_t<Delimiter>& opt_start)
{
	if (not opt_start.ever_used) {
		opt_start.ever_used = true;
	}
	else {
		mt << opt_start.delimiter_;
	}
	return mt;
}


// A partial specialization gadget
template<typename CompilationOptions, typename MarshalTarget, typename Delimiter>
struct gadget {
	/**
	 * Uses the streaming/left-shift operator (<<) to render a delimited sequence of
	 * command-line-argument-like options (with or without a value as relevant)
	 * into some target entity - which could be a buffer of chars or a more complex
	 * structure like @ref marshalled_options_t.
	 */
	static void process(
		const CompilationOptions &opts, MarshalTarget &marshalled, Delimiter delimiter,
		bool need_delimiter_after_last_option);
};


template <typename CompilationOptions, typename MarshalTarget, typename Delimiter>
void process(
	const CompilationOptions& opts, MarshalTarget& marshalled, Delimiter delimiter,
	bool need_delimiter_after_last_option = false)
{
	return detail_::gadget<CompilationOptions, MarshalTarget, Delimiter>::process(
		opts, marshalled, delimiter, need_delimiter_after_last_option);
}

} // namespace detail_

/**
 * Finalize a compilation options "building" object into a structure passable to some of the
 * CUDA JIT compilation APIs
 *
 * @tparam Kind The kind of JITable program options to render
 *
 * @return A structure of multiple strings, passable to various CUDA APIs, but no longer
 * easy to modify and manipulate.
 */
template <typename CompilationOptions>
inline detail_::marshalled_options_t marshal(const CompilationOptions& opts)
{
	using detail_::marshalled_options_t;
	marshalled_options_t marshalled;
	// TODO: Can we easily determine the max number of options here?
	enum : bool { need_delimiter_after_last_option = true };
	marshalling::detail_::process(opts, marshalled, marshalled_options_t::advance_gadget{},
		need_delimiter_after_last_option);
	return marshalled;
}

/**
 * Finalize a set of options into the form of a string appendable to a command-line
 *
 * @tparam Kind The kind of options to render
 *
 * @return a string made up of command-line options - switches and options with arguments,
 * designated by single or double dashes.
 *
 * @note An implementation of a processor/renderer of individual options must be
 * provided via detail_::process() for this function to be usable with any particular
 * type of options.
 *
 */
template <typename CompilationOptions>
inline ::std::string render(const CompilationOptions& opts)
{
	::std::ostringstream oss;
	detail_::process(opts, oss, ' ');
	if (oss.tellp() > 0) {
		// Remove the last, excessive, delimiter
		oss.seekp(-1,oss.cur);
	}
	return oss.str();
}

} // namespace marshalling

} // namespace cuda

#endif /* SRC_CUDA_API_DETAIL_OPTION_MARSHALLING_HPP_ */
