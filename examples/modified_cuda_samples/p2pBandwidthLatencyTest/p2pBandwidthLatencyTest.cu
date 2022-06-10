/**
 * Derived from the nVIDIA CUDA 10.2 samples by
 *
 *   Eyal Rozenberg
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 *
 * The original code is Copyright 1993-2015 NVIDIA Corporation.
 */

#include <vector>
#include <cstdlib>

#include "../helper_cuda.hpp"

using namespace std;

typedef enum
{
    P2P_WRITE = 0,
    P2P_READ = 1,
} P2PDataTransfer;

typedef enum
{
    CE = 0, 
    SM = 1,
}P2PEngine;

struct command_line_options {
    bool test_p2p_read;
    P2PEngine mechanism;
};

template <typename T>
struct square_matrix {
	using size_type = typename ::std::vector<T>::size_type;
	::std::vector<T> elements;
	size_type leg;

	static square_matrix make(::std::size_t leg)
	{
		square_matrix<T> result;
		result.elements.resize(leg*leg);
		result.leg = leg;
		return result;
	}

	T& operator()(size_type x, size_type y) { return elements.at(x * leg + y); }
	T operator()(size_type x, size_type y) const { return elements.at(x * leg + y); }

};

// template square_matrix<double>;

void print(const square_matrix<double> &bandwidthMatrix, const char* axis_cross_label);

constexpr const unsigned long long default_timeout_clocks = 10000000ull;

namespace bandwidth {
	constexpr const int precision { 2 };
	constexpr const int integral_digits { 3 };
	constexpr const int field_width { precision + 1 + integral_digits };
} // namespace bandwidth

__global__ void delay(volatile int *flag, unsigned long long timeout_clocks = default_timeout_clocks)
{
    // Wait until the application notifies us that it has completed queuing up the
    // experiment, or timeout and exit, allowing the application to make progress
    long long int start_clock, sample_clock;
    start_clock = clock64();

    while (!*flag) {
        sample_clock = clock64();

        if (sample_clock - start_clock > timeout_clocks) {
            break;
        }
    }
}

// This kernel is for demonstration purposes only, not a performant kernel for p2p transfers.
__global__ void copyp2p(
    int4*         __restrict__  dest,
    int4   const* __restrict__  src,
    size_t                      num_elems)
{
    size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = blockDim.x * gridDim.x;

    #pragma unroll(5)
    for (size_t i=globalId; i < num_elems; i+= gridSize)
    {
        dest[i] = src[i];
    }
}

///////////////////////////////////////////////////////////////////////////
//Print help screen
///////////////////////////////////////////////////////////////////////////
void printHelp(void)
{
    ::std::cout
        << "Usage:  p2pBandwidthLatencyTest [OPTION]...\n"
        << "Tests bandwidth/latency of GPU pairs using P2P and without P2P\n"
        << "\n"
        << "Options:\n"
        << "--help\t\tDisplay this help menu\n"
        << "--p2p_read\tUse P2P reads for data transfers between GPU pairs and show corresponding results.\n \t\tDefault used is P2P write operation.\n"
        << "--sm_copy\t\tUse SM initiated p2p transfers instead of Copy Engine\n";
}

void checkP2Paccess()
{
    for (auto device : cuda::devices()) {
    	static_assert(::std::is_same<decltype(device), cuda::device_t>::value, "Unexpected type");
        for (auto peer : cuda::devices()) {
            if ((cuda::device_t)device != (cuda::device_t)peer) {
                auto access = cuda::device::peer_to_peer::can_access(device, peer);
                ::std::cout << "Device " << device.id() << (access ? " CAN" : " CANNOT") << " access Peer Device " << peer.id() << "\n";
            }
        }
    }
    ::std::cout << "\nNote: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.\nSo you can see lesser Bandwidth (GB/s) and unstable Latency (us) in those cases.\n\n";
}

void enqueue_p2p_copy(
    int *dest,
    int *src,
    ::std::size_t num_elems,
    int repeat,
    bool p2paccess,
    P2PEngine p2p_mechanism,
    cuda::stream_t& stream)
{
#if CUDA_VERSION <= 10000
    (void) p2paccess;
    (void) p2p_mechanism;
#else
    if (p2p_mechanism == SM && p2paccess)
    {
        auto copy_kernel = cuda::kernel::get(stream.device(), copyp2p);
        auto grid_and_block_dims = copy_kernel.min_grid_params_for_max_occupancy();
        // Note: We could have alternatively used:
        // auto grid_and_block_dims = cuda::kernel::occupancy::min_grid_params_for_max_occupancy(copy_kernel);
        auto launch_config = cuda::make_launch_config(grid_and_block_dims);

        for (int r = 0; r < repeat; r++) {
            stream.enqueue.kernel_launch(copy_kernel, launch_config, (int4*)dest, (int4*)src, num_elems/sizeof(int4));
        }
    }
    else
#endif // CUDA_VERSION >= 10000
  {
        for (int r = 0; r < repeat; r++) {
        // Since we assume Compute Capability >= 2.0, all devices support the
        // Unified Virtual Address Space, so we don't need to use
        // cudaMemcpyPeerAsync - cudaMemcpyAsync is enough.
            cuda::memory::async::copy(dest, src, sizeof(*dest)*num_elems, stream);
        }
    }
}

void outputBandwidthMatrix(P2PEngine mechanism, bool test_p2p, P2PDataTransfer p2p_method)
{
    int numElems = 10000000;
    int repeat = 5;
	vector<cuda::stream_t> streams;
    vector<cuda::memory::device::unique_ptr<int[]>> buffers;
    vector<cuda::memory::device::unique_ptr<int[]>> buffersD2D; // buffer for D2D, that is, intra-GPU copy
    vector<cuda::event_t> start;
    vector<cuda::event_t> stop;

    auto flag = reinterpret_cast<volatile int *>(
        cuda::memory::host::allocate(sizeof(int), cuda::memory::portability_across_contexts::is_portable)
    );

    for (auto device : cuda::devices()) {
        streams.push_back(device.create_stream(cuda::stream::async));
        buffers.push_back(cuda::memory::device::make_unique<int[]>(device, numElems));
        buffersD2D.push_back(cuda::memory::device::make_unique<int[]>(device, numElems));
        start.push_back(device.create_event());
        stop.push_back(device.create_event());
    }

    auto num_gpus = cuda::device::count();
    auto bandwidthMatrix = square_matrix<double>::make(num_gpus);

    for (auto device : cuda::devices()) {
        for (auto peer : cuda::devices()) {
            bool p2p_access_possible { false };
            if (test_p2p) {
                p2p_access_possible = device.can_access(peer);
                if (p2p_access_possible) {
                    cuda::device::peer_to_peer::enable_access(device, peer);
                    cuda::device::peer_to_peer::enable_access(peer, device);
                }
            }

            auto i = device.id();
            auto j = peer.id();

            streams[i].synchronize();

            // Block the stream until all the work is queued up
            // DANGER! - cudaMemcpy*Async may infinitely block waiting for
            // room to push the operation, so keep the number of repeatitions
            // relatively low.  Higher repeatitions will cause the delay kernel
            // to timeout and lead to unstable results.
            *flag = 0;
            streams[i].enqueue.kernel_launch(delay, cuda::launch_configuration_t{1, 1, 0}, flag, default_timeout_clocks);
            streams[i].enqueue.event(start[i]);

            if (i == j) {
                // Perform intra-GPU, D2D copies
                enqueue_p2p_copy(buffers[i].get(), buffersD2D[i].get(), numElems, repeat, p2p_access_possible, mechanism, streams[i]);

            }
            else {
                if (p2p_method == P2P_WRITE)
                {
                    enqueue_p2p_copy(buffers[j].get(), buffers[i].get(), numElems, repeat, p2p_access_possible, mechanism, streams[i]);
                }
                else
                {
                    enqueue_p2p_copy(buffers[i].get(), buffers[j].get(), numElems, repeat, p2p_access_possible, mechanism, streams[i]);
                }
            }

            streams[i].enqueue.event(stop[i]);

            // Release the queued events
            *flag = 1;
            streams[i].synchronize();

            using usec_duration_t = ::std::chrono::duration<double, ::std::micro>;
            usec_duration_t duration ( cuda::event::time_elapsed_between(start[i], stop[i]) );

            double gb = numElems * sizeof(int) * repeat / (double)1e9;
            if (i == j) {
                gb *= 2;    //must count both the read and the write here
            }
            bandwidthMatrix(i, j) = gb / duration.count();
            if (test_p2p and p2p_access_possible) {
                cuda::device::peer_to_peer::disable_access(device, peer);
                cuda::device::peer_to_peer::disable_access(peer, device);
            }
        }
    }

    ::std::cout << "Unidirectional P2P=" << (test_p2p ? "Enabled": "Disabled") << " Bandwidth " << (p2p_method == P2P_READ ? "(P2P Reads) " : "") << "Matrix (GB/s)\n";

	print(bandwidthMatrix, "D\\D");

}

void print(const square_matrix<double> &bandwidthMatrix, const char* axis_cross_label)
{
	constexpr const ::std::size_t width = bandwidth::field_width;
	cout << ::std::setw(width) << axis_cross_label;

	for (auto device : cuda::devices()) {
		cout << setw(width) << device.id() << ' ';
	}

	cout << "\n";

	for (auto device : cuda::devices()) {
		cout << setw(width) << device.id() << ' ';

		for (auto peer : cuda::devices()) {
			cout << fixed << setprecision(bandwidth::precision) << setw(width)
			<< bandwidthMatrix(device.id() , peer.id()) << ' ';
		}

		cout << "\n";
	}
	cout << "\n";
}

void outputBidirectionalBandwidthMatrix(P2PEngine p2p_mechanism, bool test_p2p)
{
    int numElems = 10000000;
    int repeat = 5;

	// Note: The order of these declarations matters. The reason is, that memory
	// allocations do not hold a reference count to a primary context, while
	// streams do (and even that is a bit difficult to put faith in).
	// To avoid this, one would have to ensure the primary contexts are active
	// through the lifetimes of the pointers. That could mean, for example:
	//
	// 1. Holding a refunit-holding primary context wrappers array
	// 2. Placing the unique pointers within a curly-brackets scope, so they
	//    get release while the primary contexts array is still alive.
	//

	vector<cuda::stream_t> streams_0;
	vector<cuda::stream_t> streams_1;
    vector<cuda::memory::device::unique_ptr<int[]>> buffers;
    vector<cuda::memory::device::unique_ptr<int[]>> buffersD2D; // buffer for D2D, that is, intra-GPU copy
    vector<cuda::event_t> start;
    vector<cuda::event_t> stop;


    auto flag = reinterpret_cast<volatile int *>(
        cuda::memory::host::allocate(sizeof(int), cuda::memory::portability_across_contexts::is_portable)
    );

    for (auto device : cuda::devices()) {
        streams_0.push_back(device.create_stream(cuda::stream::async));
        streams_1.push_back(device.create_stream(cuda::stream::async));
        buffers.push_back(cuda::memory::device::make_unique<int[]>(device, numElems));
        buffersD2D.push_back(cuda::memory::device::make_unique<int[]>(device, numElems));
        start.push_back(device.create_event());
        stop.push_back(device.create_event());
    }

    auto num_gpus = cuda::devices().size();
    auto bandwidthMatrix = square_matrix<double>::make(num_gpus);

    for (int i = 0; i < num_gpus; i++) {
        auto device = cuda::device::get(i);

        for (int j = 0; j < num_gpus; j++) {
            auto peer = cuda::device::get(j);
            bool p2p_access_possible { false };
            if (test_p2p) {
                p2p_access_possible = device.can_access(peer);
                if (p2p_access_possible) {
                    cuda::device::peer_to_peer::enable_access(device, peer);
                    cuda::device::peer_to_peer::enable_access(peer, device);
                }
            }

            streams_0[i].synchronize();
            streams_1[i].synchronize();

            // Block the stream until all the work is queued up
            // DANGER! - cudaMemcpy*Async may infinitely block waiting for
            // room to push the operation, so keep the number of repeatitions
            // relatively low.  Higher repeatitions will cause the delay kernel
            // to timeout and lead to unstable results.
            *flag = 0;
            // No need to block stream1 since it'll be blocked on stream0's event
            streams_0[i].enqueue.kernel_launch(delay, cuda::launch_configuration_t{1, 1, 0}, flag, default_timeout_clocks);

            // Force stream1 not to start until stream0 does, in order to ensure
            // the events on stream0 fully encompass the time needed for all operations
            streams_0[i].enqueue.event(start[i]);
            streams_1[j].enqueue.wait(start[i]);

            if (i == j) {
                // For intra-GPU perform 2 memcopies buffersD2D <-> buffers
                enqueue_p2p_copy(buffers[i].get(), buffersD2D[i].get(), numElems, repeat, p2p_access_possible, p2p_mechanism, streams_0[i]);
                enqueue_p2p_copy(buffersD2D[i].get(), buffers[i].get(), numElems, repeat, p2p_access_possible, p2p_mechanism, streams_1[i]);
            }
            else {
                enqueue_p2p_copy(buffers[i].get(), buffers[j].get(), numElems, repeat, p2p_access_possible, p2p_mechanism, streams_1[j]);
                enqueue_p2p_copy(buffers[j].get(), buffers[i].get(), numElems, repeat, p2p_access_possible, p2p_mechanism, streams_0[i]);
            }

            // Notify stream0 that stream1 is complete and record the time of
            // the total transaction
            streams_1[j].enqueue.event(stop[j]);
            streams_0[i].enqueue.wait(stop[j]);
            streams_0[i].enqueue.event(stop[i]);

            // Release the queued operations
            *flag = 1;
            streams_0[i].synchronize();
            streams_1[j].synchronize();

            using seconds_duration_t = ::std::chrono::duration<double>;
            seconds_duration_t duration ( cuda::event::time_elapsed_between(start[i], stop[i]) );

            double gb = 2.0 * numElems * sizeof(int) * repeat / (double)1e9;
            if (i == j) {
                gb *= 2;    //must count both the read and the write here
            }
            bandwidthMatrix(i, j) = gb / duration.count();
            if (test_p2p && p2p_access_possible) {
                cuda::device::peer_to_peer::disable_access(device, peer);
                cuda::device::peer_to_peer::disable_access(peer, device);
            }
        }
    }

    ::std::cout << "Bidirectional P2P=" << (test_p2p ? "Enabled": "Disabled") << " Bandwidth Matrix (GB/s)\n";

    print(bandwidthMatrix, "D\\D");
}

void outputLatencyMatrix(P2PEngine p2p_mechanism, bool test_p2p, P2PDataTransfer p2p_method)
{
    int repeat = 100;
    int numElems = 4; // perform 1-int4 transfer.

	// Note: The order of these declarations matters. The reason is, that memory
	// allocations do not hold a reference count to a primary context, while
	// streams do (and even that is a bit difficult to put faith in).
	// To avoid this, one would have to ensure the primary contexts are active
	// through the lifetimes of the pointers. That could mean, for example:
	//
	// 1. Holding a refunit-holding primary context wrappers array
	// 2. Placing the unique pointers within a curly-brackets scope, so they
	//    get release while the primary contexts array is still alive.
	//

	vector<cuda::stream_t> streams;
    vector<cuda::memory::device::unique_ptr<int[]>> buffers;
    vector<cuda::memory::device::unique_ptr<int[]>> buffersD2D; // buffer for D2D, that is, intra-GPU copy
    vector<cuda::event_t> start;
    vector<cuda::event_t> stop;

    auto flag = reinterpret_cast<volatile int *>(
        cuda::memory::host::allocate(sizeof(int), cuda::memory::portability_across_contexts::is_portable)
    );

    for(auto device : cuda::devices()) {
        streams.push_back(device.create_stream(cuda::stream::async));
        buffers.push_back(cuda::memory::device::make_unique<int[]>(device, numElems));
        buffersD2D.push_back(cuda::memory::device::make_unique<int[]>(device, numElems));
        start.push_back(device.create_event());
        stop.push_back(device.create_event());
    }

    auto num_gpus = cuda::devices().size();
    auto gpuLatencyMatrix = square_matrix<double>::make(num_gpus);
    auto cpuLatencyMatrix = square_matrix<double>::make(num_gpus);

    for (auto device : cuda::devices()) {
        for (auto peer : cuda::devices()) {
            bool p2p_access_possible { false };
            if (test_p2p) {
                p2p_access_possible = device.can_access(peer);
                if (p2p_access_possible) {
                    cuda::device::peer_to_peer::enable_access(device, peer);
                    cuda::device::peer_to_peer::enable_access(peer, device);
                }
            }
            auto i = device.id();
            auto j = peer.id();

            streams[i].synchronize();

            // Block the stream until all the work is queued up
            // DANGER! - cudaMemcpy*Async may infinitely block waiting for
            // room to push the operation, so keep the number of repeatitions
            // relatively low.  Higher repeatitions will cause the delay kernel
            // to timeout and lead to unstable results.
            *flag = 0;
            auto single_thread = cuda::make_launch_config(cuda::grid::dimensions_t::point(), cuda::grid::block_dimensions_t::point());
            streams[i].enqueue.kernel_launch(delay, single_thread, flag, default_timeout_clocks);
            streams[i].enqueue.event(start[i]);

            auto time_before_copy = ::std::chrono::high_resolution_clock::now();
            if (i == j) {
                // Perform intra-GPU, D2D copies
                enqueue_p2p_copy(buffers[i].get(), buffersD2D[i].get(), numElems, repeat, p2p_access_possible, p2p_mechanism, streams[i]);
            }
            else {
                if (p2p_method == P2P_WRITE)
                {
                    enqueue_p2p_copy(buffers[j].get(), buffers[i].get(), numElems, repeat, p2p_access_possible, p2p_mechanism, streams[i]);
                }
                else
                {
                    enqueue_p2p_copy(buffers[i].get(), buffers[j].get(), numElems, repeat, p2p_access_possible, p2p_mechanism, streams[i]);
                }
            }
            auto time_after_copy = ::std::chrono::high_resolution_clock::now();
            ::std::chrono::duration<double, ::std::micro> cpu_duration_ms = time_after_copy - time_before_copy;

            streams[i].enqueue.event(stop[i]);
            // Now that the work has been queued up, release the stream
            *flag = 1;
            streams[i].synchronize();

            using usec_duration_t = ::std::chrono::duration<double, ::std::micro>;
            usec_duration_t gpu_duration ( cuda::event::time_elapsed_between(start[i], stop[i]) );

            gpuLatencyMatrix(i, j) = gpu_duration.count() / repeat;
            cpuLatencyMatrix(i, j) = cpu_duration_ms.count() / repeat;
            if (test_p2p && p2p_access_possible) {
                cuda::device::peer_to_peer::disable_access(device, peer);
                cuda::device::peer_to_peer::disable_access(peer, device);
            }
        }
    }

    ::std::cout
        << "P2P=" << (test_p2p ? "Enabled": "Disabled") << " Latency "
        << (test_p2p ? (p2p_method == P2P_READ ? "(P2P Reads) " : "(P2P Writes) ") : "")
        << "Matrix (us)\n";

    print(gpuLatencyMatrix, "GPU");
    print(cpuLatencyMatrix, "CPU");
}

//Check peer-to-peer connectivity
void print_connectivity_matrix()
{
    //Check peer-to-peer connectivity
    ::std::cout << "P2P Connectivity Matrix\n";
    ::std::cout << "   D\\D ";
    for (auto device : cuda::devices()) {
        ::std::cout << ::std::setw(bandwidth::field_width) << device.id() << ' ';
    }
    ::std::cout << "\n";
    for (auto device : cuda::devices()) {
        ::std::cout << ::std::setw(bandwidth::field_width) << device.id() << ' ';
        for (auto peer : cuda::devices()) {
            if (device != peer) {
                auto access = cuda::device::peer_to_peer::can_access(device, peer);
                ::std::cout << ::std::setw(bandwidth::field_width) << ((access) ? 1 : 0);
            } else {
                ::std::cout << ::std::setw(bandwidth::field_width) << 1;
            }
            ::std::cout << ' ';
        }
        ::std::cout << "\n";
    }
}

void list_devices()
{
    //output devices
    for (auto device : cuda::devices()) {
        auto properties = device.properties();
        ::std::cout << "Device: " << device.id() << ", " << properties.name << ", PCI location " << properties.pci_id() << '\n';
    }
}

command_line_options handle_command_line(int argc, char** argv)
{
    P2PEngine p2p_mechanism { CE }; // By default use Copy Engine
    bool test_p2p_read { false };

    if (checkCmdLineFlag(argc, (const char**) (argv), "help")) {
        printHelp();
        exit(EXIT_SUCCESS);
    }
    if (checkCmdLineFlag(argc, (const char**) (argv), "test_p2p_read")) {
        test_p2p_read = P2P_READ;
    }
    if (checkCmdLineFlag(argc, (const char**) (argv), "sm_copy")) {
#if CUDA_VERSION <= 10000
        ::std::cerr << "This mechanism is unsupported by this program before CUDA 10.0" << ::std::endl;
        exit(EXIT_FAILURE);
#else
        p2p_mechanism = SM;
#endif
    }
    return { test_p2p_read, p2p_mechanism };
}

int main(int argc, char **argv)
{
    command_line_options opts = handle_command_line(argc, argv);
    ::std::cout << "[P2P (Peer-to-Peer) GPU Bandwidth Latency Test]\n";

    list_devices();
    checkP2Paccess();

    print_connectivity_matrix();

    enum : bool {
        do_measure_p2p = true,
        dont_measure_p2p = false
    };

    outputBandwidthMatrix(opts.mechanism, dont_measure_p2p, P2P_WRITE);
    outputBandwidthMatrix(opts.mechanism, do_measure_p2p, P2P_WRITE);
    if (opts.test_p2p_read)
    {
        outputBandwidthMatrix(opts.mechanism, do_measure_p2p, P2P_READ);
    }
    outputBidirectionalBandwidthMatrix(opts.mechanism, dont_measure_p2p);
    outputBidirectionalBandwidthMatrix(opts.mechanism, do_measure_p2p);

    outputLatencyMatrix(opts.mechanism, dont_measure_p2p, P2P_WRITE);
    outputLatencyMatrix(opts.mechanism, do_measure_p2p, P2P_WRITE);
    if (opts.test_p2p_read)
    {
        outputLatencyMatrix(opts.mechanism, do_measure_p2p, P2P_READ);
    }
    ::std::cout << "\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n";
}
