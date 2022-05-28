/**
 * An example program for the CUDA API wrappers library,
 * which indirectly manipulates the current device using
 * driver API calls.
 *
 */
#include "../common.hpp"
#include <iostream>

void report_current_device()
{
	::std::cout << "Runtime believes the current device index is: "
		<< cuda::device::current::detail_::get_id() << ::std::endl;
}

int main()
{
	namespace context = cuda::context::detail_;
	namespace cur_dev = cuda::device::current::detail_;
	namespace pc = cuda::device::primary_context::detail_;
	namespace cur_ctx = cuda::context::current::detail_;

	cuda::device::id_t dev_idx[2];
	cuda::context::handle_t pc_handle[2];
	
	cuda::initialize_driver();
	dev_idx[0] = cur_dev::get_id();
	report_current_device();
	assert_(cur_dev::get_id() == 0);
	dev_idx[1] = (dev_idx[0] == 0) ? 1 : 0;
	pc_handle[0] = pc::obtain_and_increase_refcount(dev_idx[0]);
	::std::cout << "Obtained primary context handle for device " << dev_idx[0]<< '\n';
	pc_handle[1] = pc::obtain_and_increase_refcount(dev_idx[1]);
	::std::cout << "Obtained primary context handle for device " << dev_idx[1]<< '\n';
	report_current_device();
	cur_ctx::push(pc_handle[1]);
	::std::cout << "Pushed primary context handle for device " << dev_idx[1] << " onto the stack\n";
	report_current_device();
	assert_(cur_dev::get_id() == dev_idx[1]);
	auto ctx = context::create_and_push(dev_idx[0]);
	::std::cout << "Created a new context for device " << dev_idx[0] << " and pushed it onto the stack\n";
	report_current_device();
	assert_(cur_dev::get_id() == dev_idx[0]);
	cur_ctx::push(ctx);
	::std::cout << "Pushed primary context handle for device " << dev_idx[0] << " onto the stack\n";
	report_current_device();
	assert_(cur_dev::get_id() == dev_idx[0]);
	cur_ctx::push(pc_handle[1]);
	::std::cout << "Pushed primary context for device " << dev_idx[1] << " onto the stack\n";
	report_current_device();
	assert_(cur_dev::get_id() == dev_idx[1]);
	pc::decrease_refcount(dev_idx[1]);
	::std::cout << "Deactivated/destroyed primary context for device " << dev_idx[1] << '\n';
	report_current_device();
	assert_(cur_dev::get_id() == dev_idx[1]);

	::std::cout << "\nSUCCESS" << ::std::endl;
}

