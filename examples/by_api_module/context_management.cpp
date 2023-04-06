/**
 * An example program utilizing most/all calls from the CUDA
 * Driver API module:
 *
 *   Device Management
 */
#include "../common.hpp"

void current_context_manipulation(const cuda::device_t &device, const cuda::device::primary_context_t &pc,
	const cuda::context_t &created_context);

void test_context(
	const cuda::context_t& context,
	bool is_primary,
	cuda::device::id_t device_id)
{
	std::cout << "Testing " << (is_primary ? "" : "non-") << "primary context " << context << '\n';
	if (context.device_id() != device_id) {
		die_("The device's primary context's reported ID and the device wrapper's ID differ: "
			+ std::to_string(context.device_id()) + " !=" +  std::to_string(device_id));
	}

	if (context.device().id() != device_id) {
		die_("The context's associated device's ID is not the same as that of the device for which we obtained the context: "
			+ std::to_string(context.device().id()) + " !=" +  std::to_string(device_id) );
	}

	if (context.is_primary() != is_primary) {
		die_(std::string("The ") + (is_primary ? "" : "non-") + "primary context " + std::to_string(context)
			+ " \"believes\" it is " + (is_primary ? "not " : "") + "primary.");
	}

	// Specific attributes and properties with their own API calls:
	// L1/shared mem (CacheConfig), shared memory bank size (SharedMemConfig)
	// and stream priority range
	// ----------------------------------------------------------------

	auto cache_preference = context.cache_preference();
	std::cout << "The cache preference for context " << context << " is: " << cache_preference << ".\n";

	auto new_cache_preference =
		cache_preference == cuda::multiprocessor_cache_preference_t::prefer_l1_over_shared_memory ?
		cuda::multiprocessor_cache_preference_t::prefer_shared_memory_over_l1 :
		cuda::multiprocessor_cache_preference_t::prefer_l1_over_shared_memory;
	context.set_cache_preference(new_cache_preference);
	cache_preference = context.cache_preference();
	assert_(cache_preference == new_cache_preference);
	std::cout << "The cache preference for context " << context << " has now been set to: " << new_cache_preference << ".\n";

	auto shared_mem_bank_size = context.shared_memory_bank_size();
	shared_mem_bank_size =
		(shared_mem_bank_size == CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE) ?
			CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE : CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE;
	context.set_shared_memory_bank_size(shared_mem_bank_size);
	auto stream_priority_range = context.stream_priority_range();
	if (stream_priority_range.is_trivial()) {
		std::cout << "Context " << context <<  " does not support stream priorities. "
			"All streams will have the same (default) priority.\n";
	}
	else {
		std::cout << "Streams in context " << context << " have priorities between "
			<< stream_priority_range.least << " (highest numeric value, least prioritized) and "
			<< std::to_string(stream_priority_range.greatest) << "(lowest numeric values, most prioritized).\n";
		assert(stream_priority_range.least > stream_priority_range.greatest);
	}

	// Resource limits
	// --------------------

	auto printf_fifo_size = context.get_limit(CU_LIMIT_PRINTF_FIFO_SIZE);
	std::cout << "The printf FIFO size for context " << context << " is " << printf_fifo_size << ".\n";
	decltype(printf_fifo_size) new_printf_fifo_size =
		(printf_fifo_size <= 1024) ?  2 * printf_fifo_size : printf_fifo_size - 512;
	context.set_limit(CU_LIMIT_PRINTF_FIFO_SIZE, new_printf_fifo_size);
	printf_fifo_size = context.get_limit(CU_LIMIT_PRINTF_FIFO_SIZE);
	assert_(printf_fifo_size == new_printf_fifo_size);

	// Flags - yes, yet another kind of attribute/property
	// ----------------------------------------------------

	std::cout << "Context " << context << " uses a"
		<< (context.synch_scheduling_policy() ? " synchronous" : "n asynchronous")
		<< " scheduling policy.\n";
	std::cout << "Context " << context << " is set to "
		<< (context.keeping_larger_local_mem_after_resize() ? "keep" : "discard")
		<< " shared memory allocation after launch.\n";
	// TODO: Change the settings as well obtaining them

}

void current_context_manipulation(
	cuda::device_t &device,
	cuda::device::primary_context_t &pc,
	cuda::context_t &created_context)
{
	cuda::context_t context_0 = pc;
	cuda::context_t context_1 = created_context;
	cuda::context::current::set(context_0);
	assert_(cuda::context::current::get() == context_0);
	assert_(cuda::context::current::detail_::get_handle() == context_0.handle());
	cuda::context::current::set(context_1);
	assert_(cuda::context::current::get() == context_1);
	assert_(cuda::context::current::detail_::get_handle() == context_1.handle());


	auto context_2 = cuda::context::create(device);
	{
		cuda::context::current::scoped_override_t context_for_this_block { context_2 };
		assert_(context_2.handle() == cuda::context::current::get().handle());
		assert_(context_2 == cuda::context::current::get());
	}
	auto gotten = cuda::context::current::get();
	assert_(gotten == context_1);

	auto context_3 = cuda::context::create_and_push(device);

//	std::cout << "Contexts:\n";
//	std::cout << "context_0: " << context_0 << '\n';
//	std::cout << "context_1: " << context_1 << '\n';
//	std::cout << "context_2: " << context_2 << '\n';
//	std::cout << "context_3: " << context_3 << '\n';

	{
		cuda::context::current::scoped_override_t context_for_this_block { context_3 };
		assert_(context_3.handle() == cuda::context::current::get().handle());
		assert_(context_3 == cuda::context::current::get());
	}

	{
		auto popped = cuda::context::current::pop();
		assert_(popped == context_3);
	}
	gotten = cuda::context::current::get();
	assert_(gotten == context_1);
}


int main(int argc, char **argv)
{
	auto device_id = choose_device(argc, argv);
	auto device = cuda::device::get(device_id);

	if (cuda::device::primary_context::is_active(device)) {
		std::ostringstream oss;
		oss << "The primary context is unexpectedly active before we've done anything with its device (" << device << ")\n";
		die_(oss.str());
	}

	auto original_pc = device.primary_context();

	cuda::device::primary_context::detail_::decrease_refcount(device.id());

	if (cuda::device::primary_context::is_active(device)) {
		die_("The primary context is unexpectedly active after increasing, then decreasing, its refcount");
	}

	auto pc = device.primary_context();

	// std::cout << "New PC handle = " << pc.handle() << " ; old PC handle = " << original_pc.handle() << "\n";

	cuda::device::primary_context::detail_::increase_refcount(device.id());

	cuda::context::current::push(pc);
	constexpr const bool is_primary = true;
	constexpr const bool isnt_primary = false;
	test_context(pc, is_primary, device_id);

	{
		auto popped = cuda::context::current::pop();
		if (popped != pc) {
			die_("After pushing context " + std::to_string(pc) + " and popping it - the pop result is a different context, " + std::to_string(popped));
		}
	}

	auto created_context = cuda::context::create(device);
	test_context(created_context, isnt_primary, device_id);
	current_context_manipulation(device, pc, created_context);

	std::cout << std::endl;
//	report_context_stack("After current_context_manipulation");
	cuda::context::current::push(created_context);
	cuda::context::current::push(created_context);
	// We should have 3 copies of created_context on the stack at this point, and nothing else
	cudaSetDevice(device_id);
//	report_context_stack("After cudaSetDevice " + std::to_string(device_id));
	// We should have the primary context of the device


	device.synchronize();
	device.reset();

	std::cout << "\nSUCCESS\n";
}
