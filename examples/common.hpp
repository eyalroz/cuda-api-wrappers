/**
 * @file
 *
 * @brief Common header for many/most/all CUDA API wrapper example programs.
 */
#ifndef EXAMPLES_COMMON_HPP_
#define EXAMPLES_COMMON_HPP_


#include <string>
#include <iostream>

void report_current_context(const ::std::string& prefix);
void report_context_stack(const ::std::string& prefix);

#include <cuda/api.hpp>

#include <cstdio>
#include <fstream>
#include <cmath>
#include <cstring>
#include <system_error>
#include <memory>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <numeric>

const char* cache_preference_name(cuda::multiprocessor_cache_preference_t pref)
{
	static const char* cache_preference_names[] = {
		"No preference",
		"Equal L1 and shared memory",
		"Prefer shared memory over L1",
		"Prefer L1 over shared memory",
	};
	return cache_preference_names[(off_t) pref];
}

const char* host_thread_synch_scheduling_policy_name(cuda::context::host_thread_synch_scheduling_policy_t policy)
{
	static const char *names[] = {
		"heuristic",
		"spin",
		"yield",
		"INVALID",
		"block",
		nullptr
	};
	return names[(off_t) policy];
}

const char* memory_type_name(cuda::memory::type_t mem_type)
{
	static const char* memory_type_names[] = {
		"N/A",
		"host",
		"device",
		"array",
		"unified"
	};
	return memory_type_names[mem_type];
}

namespace std {

::std::ostream& operator<<(::std::ostream& os, cuda::device::compute_capability_t cc)
{
    return os << cc.major() << '.' << cc.minor();
}

::std::ostream& operator<<(::std::ostream& os, cuda::multiprocessor_cache_preference_t pref)
{
	return (os << cache_preference_name(pref));
}

::std::ostream& operator<<(::std::ostream& os, cuda::context::host_thread_synch_scheduling_policy_t pref)
{
	return (os << host_thread_synch_scheduling_policy_name(pref));
}

::std::ostream& operator<<(::std::ostream& os, cuda::context::handle_t handle)
{
	return (os << cuda::detail_::ptr_as_hex(handle));
}

::std::ostream& operator<<(::std::ostream& os, const cuda::context_t& context)
{
	return os << "[device " << context.device_id() << " handle " << context.handle() << ']';
}

::std::ostream& operator<<(::std::ostream& os, const cuda::device_t& device)
{
	return os << cuda::device::detail_::identify(device.id());
}

::std::ostream& operator<<(::std::ostream& os, const cuda::stream_t& stream)
{
	return os << cuda::stream::detail_::identify(stream.handle(), stream.device().id());
}


::std::string to_string(const cuda::context_t& context)
{
	::std::stringstream ss;
	ss.clear();
	ss << context;
	return ss.str();
}

} // namespace std

[[noreturn]] bool die_(const ::std::string& message)
{
	::std::cerr << message << "\n";
	exit(EXIT_FAILURE);
}

#define assert_(cond) \
{ \
	auto evaluation_result = (cond); \
	if (not evaluation_result) \
		die_("Assertion failed at line " + ::std::to_string(__LINE__) + ": " #cond); \
}


void report_current_context(const ::std::string& prefix = "")
{
	if (not prefix.empty()) { ::std::cout << prefix << ", the current context is: "; }
	else ::std::cout << "The current context is: ";
	if (not cuda::context::current::exists()) {
		::std::cout << "(None)" << ::std::endl;
	}
	else {
		auto cc = cuda::context::current::get();
		::std::cout << cc << ::std::endl;
	}
}

void print_context_stack()
{
	if (not cuda::context::current::exists()) {
		::std::cout << "(Context stack is empty/uninitialized)" << ::std::endl;
		return;
	}
	::std::vector<cuda::context::handle_t> contexts;
	while(cuda::context::current::exists()) {
		contexts.push_back(cuda::context::current::detail_::pop());
	}
//	::std::cout << "" << contexts.size() << " contexts; top to bottom:\n";
	for (auto handle : contexts) {
		auto device_id = cuda::context::detail_::get_device_id(handle);
		::std::cout << handle << " for device " << device_id;
		if (cuda::context::detail_::is_primary(handle)) {
			::std::cout << " (primary, "
				<< (cuda::device::primary_context::detail_::is_active(device_id) ? "active" : "inactive")
				<< ')';
		}
		::std::cout << '\n';
	}
	for (auto it = contexts.rbegin(); it != contexts.rend(); it++) {
		cuda::context::current::detail_::push(*it);
	}
}

void report_primary_context_activity(const ::std::string& prefix = "")
{
	if (not prefix.empty()) { ::std::cout << prefix << ", "; }
	::std::cout << "Device primary contexts activity: ";
	for(auto device : cuda::devices()) {
		::std::cout << device.id() << ": "
				  << (cuda::device::primary_context::detail_::is_active(device.id()) ? "ACTIVE" : "inactive")
				  << "  ";
	}
	::std::cout << '\n';
}

void report_context_stack(const ::std::string& prefix = "")
{
	if (not prefix.empty()) { ::std::cout << prefix << ", the context stack is (top to bottom):\n"; }
	::std::cout << "-----------------------------------------------------\n";
	print_context_stack();
	::std::cout << "---\n";
	report_primary_context_activity();
	::std::cout << "-----------------------------------------------------\n" << ::std::flush;
}


// Note: This will only work correctly for positive values
template <typename U1, typename U2>
typename ::std::common_type<U1,U2>::type div_rounding_up(U1 dividend, U2 divisor)
{
	return dividend / divisor + !!(dividend % divisor);
}

#endif // EXAMPLES_COMMON_HPP_
