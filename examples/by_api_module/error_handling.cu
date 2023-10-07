/**
 * An example program utilizing most/all calls
 * from the CUDA Runtime API module:
 *
 *   Error Handling
 *
 */
#include "../common.hpp"

using std::cout;
using std::cerr;
using std::flush;

int main(int, char **)
{
	auto device_count = cuda::device::count();
	if (device_count == 0) {
		die_("No CUDA devices on this system");
	}

	try {
		cuda::device::current::detail_::set(device_count);
		die_("An exception should have be thrown when setting the current device to one-past-the-last.");
	}
	catch(cuda::runtime_error& e) {
		if (e.code() != cuda::status::invalid_device) { throw e; }
	}
	try {
		cuda::outstanding_error::ensure_none();
	}
	catch(cuda::runtime_error&) {
		die_("An error was outstanding, despite our not having committed any 'sticky' errors)");
	}

	cuda::device::current::set(cuda::device::get(0));
	auto device = cuda::device::current::get();

	bool got_expected_exception = false;
	try {
		cuda::launch_configuration_t lc = cuda::launch_config_builder()
			.overall_size(2048)
			.block_dimensions(15000) // Note: higher than the possible maximum for know CUDA devices
			.build();
		(void) lc;
	} catch (::std::invalid_argument& ex) {
		got_expected_exception = true;
	}
	if (not got_expected_exception) {
		die_("Should have gotten an ::std::invalid_argument exception about a launch configuration, but - didn't");
	}

	std::cout << "SUCCESS\n";
	return EXIT_SUCCESS;
}
