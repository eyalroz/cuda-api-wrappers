#include "cuda/api_wrappers.h"

#include <iostream>
#include <string>

using std::cout;
using std::cerr;
using std::flush;

int main(int argc, char **argv)
{
	auto device_count = cuda::device::count();
	if (device_count > 1) {
		cout << "There are " << device_count << " CUDA devices on this system.\n";
	}
	else {
		cout << "There is a single CUDA device on this system.\n";
	}
	cout
		<< "Will now trying to set current device ID to " << device_count
		<< " (that is, higher than the highest existing device's ID); doing that "
		<< "should throw an exception.\n";
	cout << "\n\tcuda::device::current::set(" << device_count << ")... " << flush;

	try {
		cuda::device::current::set(device_count);
		cerr << "Expected an exception to be thrown - but it wasn't!\n";
		exit(EXIT_FAILURE);
	}
	catch(cuda::runtime_error& e) {
		cout
			<< "and caught an exception regarding a CUDA runtime error.\n\n"
			<< "The exception's associated CUDA error code is " << e.error_code() << ".\n"
			<< "The exception's what() string is \"" << e.what() << "\".\n";
	}


	cout
		<< "\n"
		<< "Let us now attempt to ensure "
		<< "there are no outstanding errors. This should fail, by throwing an exception, since we _do_ have an outstanding "
		<< "error - our earlier attempt to set an invalid device.\n";

	cout << "\n\tcuda::ensure_no_outstanding_error()... " << flush;
	try {
		cuda::ensure_no_outstanding_error();
		cerr
			<< "Expected an exception to be thrown by cuda::ensure_no_outstanding_error(), "
			<< "since there _had_ been an outstanding error - but an exception wasn't thrown!\n";
		exit(EXIT_FAILURE);
	}
	catch(cuda::runtime_error& e) {
		cout
			<< "a CUDA runtime error exception was thrown.\n\n"
			<< "The exception's associated CUDA error code is " << e.error_code() << "; "
			<< "its what() string is \"" << e.what() << "\".\n";
	}


	cout
		<< "\n"
		<< "Let's try ensuring there are no exceptions one more time. Our last invocation of "
		<< "cuda::ensure_no_outstanding_error() should have cleared any outstanding errors, so "
		<< "this time it should _not_ throw an exception.\n";
	cout << "\n\tcuda::ensure_no_outstanding_error()... " << flush;
	cuda::ensure_no_outstanding_error();
	cout << "no exception thrown.\n\n";


	cout
		<< "We will now repeat all three calls (invalid setting and ensure_no_outstanding_error twice), "
		<< "but this time we'll tell ensure_no_outstanding_error() not to clear any outstanding error; "
		<< "we should trigger an exception with all three calls.\n";

	cout << "\n\tcuda::device::current::set(" << device_count << ")... " << flush;
	try {
		cuda::device::current::set(device_count);
		cerr << "Expected an exception to be thrown - but it wasn't!\n";
		exit(EXIT_FAILURE);
	}
	catch(cuda::runtime_error& e) {
		cout << e.what() << "\n";
	}

	cout << "\tcuda::ensure_no_outstanding_error(cuda::errors::dont_clear)... " << flush;
	try {
		cuda::ensure_no_outstanding_error(cuda::errors::dont_clear);
		cerr << "Expected an exception to be thrown - but it wasn't!\n";
		exit(EXIT_FAILURE);
	}
	catch(cuda::runtime_error& e) {
		cout << e.what() << "\n";
	}

	cout << "\tcuda::ensure_no_outstanding_error(cuda::errors::dont_clear)... " << flush;
	try {
		cuda::ensure_no_outstanding_error(cuda::errors::dont_clear);
		cerr << "Expected an exception to be thrown - but it wasn't!\n";
		exit(EXIT_FAILURE);
	}
	catch(cuda::runtime_error& e) {
		cout << e.what() << "\n\n";
	}
	std::cout << "SUCCESS\n";
	return EXIT_SUCCESS;
}
