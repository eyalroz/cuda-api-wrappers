/**
 * Derived from the nVIDIA CUDA 10.2 samples by
 *
 *   Eyal Rozenberg <eyalroz@technion.ac.il>
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

#include "../helper_cuda.h"
#include "../helper_timer.h"

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


constexpr const unsigned long long default_timeout_clocks = 10000000ull;

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
    std::cout
        << "Usage:  p2pBandwidthLatencyTest [OPTION]...\n"
        << "Tests bandwidth/latency of GPU pairs using P2P and without P2P\n"
        << "\n"
        << "Options:\n"
        << "--help\t\tDisplay this help menu\n"
        << "--p2p_read\tUse P2P reads for data transfers between GPU pairs and show corresponding results.\n \t\tDefault used is P2P write operation.\n"
        << "--sm_copy\t\tUse SM initiated p2p transfers instead of Copy Engine\n";
}

void checkP2Paccess(int numGPUs)
{
    for (int i = 0; i < numGPUs; i++) {
        auto device = cuda::device::get(i);
        for (int j = 0; j < numGPUs; j++) {
            auto peer = cuda::device::get(j);
            if (i != j) {
                auto access = cuda::device::peer_to_peer::can_access(device, peer);
                std::cout << "Device=" << device.id() << (access ? "CAN" : "CANNOT") << " access Peer Device " << peer.id() << "\n";
            }
        }
    }
    std::cout << "\n***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.\nSo you can see lesser Bandwidth (GB/s) and unstable Latency (us) in those cases.\n\n";
}

void enqueue_p2p_copy(
    int *dest,
    int *src,
    std::size_t num_elems,
    int repeat,
    bool p2paccess,
    P2PEngine p2p_mechanism,
    cuda::stream_t& stream)
{
    auto copy_kernel = cuda::kernel_t(stream.device(), copyp2p);
    auto params = copy_kernel.min_grid_params_for_max_occupancy();
    auto launch_config = cuda::make_launch_config(params.first, params.second);


    if (p2p_mechanism == SM && p2paccess)
    {
        for (int r = 0; r < repeat; r++) {
            stream.enqueue.kernel_launch(copy_kernel, launch_config, (int4*)dest, (int4*)src, num_elems/sizeof(int4));
        }
    }
    else
    {
        for (int r = 0; r < repeat; r++) {
        // Since we assume Compute Capability >= 2.0, all devices support the
        // Unified Virtual Address Space, so we don't need to use
        // cudaMemcpyPeerAsync - cudaMemcpyAsync is enough.
            cuda::memory::async::copy(dest, src, sizeof(*dest)*num_elems, stream);
        }
    }
}

void outputBandwidthMatrix(P2PEngine mechanism, int numGPUs, bool test_p2p, P2PDataTransfer p2p_method)
{
    int numElems = 10000000;
    int repeat = 5;
    vector<cuda::memory::device::unique_ptr<int[]>> buffers;
    vector<cuda::memory::device::unique_ptr<int[]>> buffersD2D; // buffer for D2D, that is, intra-GPU copy
    vector<cuda::event_t> start;
    vector<cuda::event_t> stop;
    vector<cuda::stream_t> streams;

    auto flag = reinterpret_cast<volatile int *>(
    	cuda::memory::host::allocate(sizeof(int), cuda::memory::portability_across_contexts::is_portable)
    );

    for (int d = 0; d < numGPUs; d++) {
        auto device = cuda::device::get(d);
        streams.push_back(device.create_stream(cuda::stream::async));
        buffers.push_back(cuda::memory::device::make_unique<int[]>(device, numElems));
        buffersD2D.push_back(cuda::memory::device::make_unique<int[]>(device, numElems));
        start.push_back(device.create_event());
        stop.push_back(device.create_event());
    }

    vector<double> bandwidthMatrix(numGPUs * numGPUs);

    for (int i = 0; i < numGPUs; i++) {
        auto device = cuda::device::get(i);

        for (int j = 0; j < numGPUs; j++) {
            auto peer = cuda::device::get(j);
            bool p2p_access_possible { false };
            if (test_p2p) {
                p2p_access_possible = device.can_access(peer);
                if (p2p_access_possible) {
                    cuda::device::peer_to_peer::enable_access(device, peer);
                    cuda::device::peer_to_peer::enable_access(peer, device);
                }
            }

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

            using seconds_duration_t = std::chrono::duration<double>;
            seconds_duration_t duration ( cuda::event::time_elapsed_between(start[i], stop[i]) );

            double gb = numElems * sizeof(int) * repeat / (double)1e9;
            if (i == j) {
                gb *= 2;    //must count both the read and the write here
            }
            bandwidthMatrix[i * numGPUs + j] = gb / duration.count();
            if (test_p2p and p2p_access_possible) {
                cuda::device::peer_to_peer::disable_access(device, peer);
                cuda::device::peer_to_peer::disable_access(peer, device);
            }
        }
    }

    std::cout << "Unidirectional P2P=" << (test_p2p ? "Enabled": "Disabled") << " Bandwidth " << (p2p_method == P2P_READ ? "(P2P Reads)" : "") << " Matrix (GB/s)\n";

    std::cout << "   D\\D";

    for (int j = 0; j < numGPUs; j++) {
        std::cout << std::setw(6) << j << ' ';
    }

    std::cout << "\n";

    for (int i = 0; i < numGPUs; i++) {
        std::cout << std::setw(6) << i << ' ';

        for (int j = 0; j < numGPUs; j++) {
            std::cout << std::setprecision(2) << std::setw(9) << bandwidthMatrix[i * numGPUs + j];
        }

        std::cout << "\n";
    }
}

void outputBidirectionalBandwidthMatrix(P2PEngine p2p_mechanism, int numGPUs, bool test_p2p)
{
    int numElems = 10000000;
    int repeat = 5;
    vector<cuda::memory::device::unique_ptr<int[]>> buffers;
    vector<cuda::memory::device::unique_ptr<int[]>> buffersD2D; // buffer for D2D, that is, intra-GPU copy
    vector<cuda::event_t> start;
    vector<cuda::event_t> stop;
    vector<cuda::stream_t> streams_0;
    vector<cuda::stream_t> streams_1;


    auto flag = reinterpret_cast<volatile int *>(
    	cuda::memory::host::allocate(sizeof(int), cuda::memory::portability_across_contexts::is_portable)
    );

    for (int d = 0; d < numGPUs; d++) {
        auto device = cuda::device::get(d);
        streams_0.push_back(device.create_stream(cuda::stream::async));
        streams_1.push_back(device.create_stream(cuda::stream::async));
        buffers.push_back(cuda::memory::device::make_unique<int[]>(device, numElems));
        buffersD2D.push_back(cuda::memory::device::make_unique<int[]>(device, numElems));
        start.push_back(device.create_event());
        stop.push_back(device.create_event());
    }

    vector<double> bandwidthMatrix(numGPUs * numGPUs);

    for (int i = 0; i < numGPUs; i++) {
        auto device = cuda::device::get(i);

        for (int j = 0; j < numGPUs; j++) {
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
            streams_0[j].enqueue.wait(stop[j]);
            streams_0[j].enqueue.event(stop[i]);

            // Release the queued operations
            *flag = 1;
            streams_0[i].synchronize();
            streams_1[j].synchronize();

            using seconds_duration_t = std::chrono::duration<double>;
            seconds_duration_t duration ( cuda::event::time_elapsed_between(start[i], stop[i]) );

            double gb = 2.0 * numElems * sizeof(int) * repeat / (double)1e9;
            if (i == j) {
                gb *= 2;    //must count both the read and the write here
            }
            bandwidthMatrix[i * numGPUs + j] = gb / duration.count();
            if (test_p2p && p2p_access_possible) {
                cuda::device::peer_to_peer::disable_access(device, peer);
                cuda::device::peer_to_peer::disable_access(peer, device);
            }
        }
    }

    std::cout << "Bidirectional P2P=" << (test_p2p ? "Enabled": "Disabled") << " Bandwidth Matrix (GB/s)\n";

    std::cout << "   D\\D";

    for (int j = 0; j < numGPUs; j++) {
        std::cout << std::setw(6) << j << ' ';
    }

    std::cout << "\n";

    for (int i = 0; i < numGPUs; i++) {
        std::cout << std::setw(6) << i << ' ';

        for (int j = 0; j < numGPUs; j++) {
            std::cout << std::setprecision(2) << std::setw(9) << bandwidthMatrix[i * numGPUs + j];
        }

        std::cout << "\n";
    }
}

void outputLatencyMatrix(P2PEngine p2p_mechanism, int numGPUs, bool test_p2p, P2PDataTransfer p2p_method)
{
    int repeat = 100;
    int numElems = 4; // perform 1-int4 transfer.
    StopWatchInterface *stopWatch = NULL;
    vector<cuda::memory::device::unique_ptr<int[]>> buffers;
    vector<cuda::memory::device::unique_ptr<int[]>> buffersD2D; // buffer for D2D, that is, intra-GPU copy
    vector<cuda::event_t> start;
    vector<cuda::event_t> stop;
    vector<cuda::stream_t> streams;

    auto flag = reinterpret_cast<volatile int *>(
    	cuda::memory::host::allocate(sizeof(int), cuda::memory::portability_across_contexts::is_portable)
    );

    // Note: The following uses code in helper_cuda, which obviously isn't covered by the API wrappers. It is therefore ugly...
    if (!sdkCreateTimer(&stopWatch)) {
        std::cout << "Failed to create stop watch\n";
        exit(EXIT_FAILURE);
    }
    sdkStartTimer(&stopWatch);

    for (int d = 0; d < numGPUs; d++) {
        auto device = cuda::device::get(d);
        streams.push_back(device.create_stream(cuda::stream::async));
        buffers.push_back(cuda::memory::device::make_unique<int[]>(device, numElems));
        buffersD2D.push_back(cuda::memory::device::make_unique<int[]>(device, numElems));
        start.push_back(device.create_event());
        stop.push_back(device.create_event());
    }

    vector<double> gpuLatencyMatrix(numGPUs * numGPUs);
    vector<double> cpuLatencyMatrix(numGPUs * numGPUs);

    for (int i = 0; i < numGPUs; i++) {
        auto device = cuda::device::get(i);

        for (int j = 0; j < numGPUs; j++) {
            auto peer = cuda::device::get(i);
            bool p2p_access_possible { false };
            if (test_p2p) {
                p2p_access_possible = device.can_access(peer);
                if (p2p_access_possible) {
                    cuda::device::peer_to_peer::enable_access(device, peer);
                    cuda::device::peer_to_peer::enable_access(peer, device);
                }
            }
            streams[i].synchronize();

            // Block the stream until all the work is queued up
            // DANGER! - cudaMemcpy*Async may infinitely block waiting for
            // room to push the operation, so keep the number of repeatitions
            // relatively low.  Higher repeatitions will cause the delay kernel
            // to timeout and lead to unstable results.
            *flag = 0;
            auto single_thread = cuda::make_launch_config(cuda::grid::dimensions_t::point(), cuda::grid::dimensions_t::point());
            streams[i].enqueue.kernel_launch(delay, single_thread, flag, default_timeout_clocks);
            streams[i].enqueue.event(start[i]);

            sdkResetTimer(&stopWatch);
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
            float cpu_time_ms = sdkGetTimerValue(&stopWatch);

            streams[i].enqueue.event(stop[i]);
            // Now that the work has been queued up, release the stream
            *flag = 1;
            streams[i].synchronize();

            using seconds_duration_t = std::chrono::duration<float>;
            seconds_duration_t duration ( cuda::event::time_elapsed_between(start[i], stop[i]) );

            gpuLatencyMatrix[i * numGPUs + j] = duration.count() / repeat;
            cpuLatencyMatrix[i * numGPUs + j] = duration.count() / repeat;
            if (test_p2p && p2p_access_possible) {
                cuda::device::peer_to_peer::disable_access(device, peer);
                cuda::device::peer_to_peer::disable_access(peer, device);
            }
        }
    }

    std::cout
        << "P2P=" << (test_p2p ? "Enabled": "Disabled") << " Latency "
        << (test_p2p ? (p2p_method == P2P_READ ? "(P2P Reads) " : "(P2P Writes)") : "")
        << " Matrix (us)\n";

    std::cout << "   GPU";

    for (int j = 0; j < numGPUs; j++) {
        std::cout << std::setw(6) << j << ' ';
    }

    std::cout << "\n";

    for (int i = 0; i < numGPUs; i++) {
        std::cout << std::setw(6) << i << ' ';

        for (int j = 0; j < numGPUs; j++) {
            std::cout << std::setprecision(2) << std::setw(9) << gpuLatencyMatrix[i * numGPUs + j];
        }

        std::cout << "\n";
    }

    std::cout << "\n   CPU";

    for (int j = 0; j < numGPUs; j++) {
        std::cout << std::setw(6) << j << ' ';
    }

    std::cout << "\n";

    for (int i = 0; i < numGPUs; i++) {
        std::cout << std::setw(6) << i << ' ';

        for (int j = 0; j < numGPUs; j++) {
            std::cout << std::setprecision(2) << std::setw(9) << cpuLatencyMatrix[i * numGPUs + j];
        }

        std::cout << "\n";
    }

    sdkDeleteTimer(&stopWatch);
}

//Check peer-to-peer connectivity
void print_connectivity_matrix(const cuda::device::id_t numGPUs)
{
    //Check peer-to-peer connectivity
    std::cout << "P2P Connectivity Matrix\n";
    std::cout << "     D\\D";
    for (int j = 0; j < numGPUs; j++) {
        std::cout << std::setw(6) << j << ' ';
    }
    std::cout << "\n";
    for (int i = 0; i < numGPUs; i++) {
        std::cout << std::setw(6) << i << '\t';
        auto device = cuda::device::get(i);
        for (int j = 0; j < numGPUs; j++) {
            if (i != j) {
                auto peer = cuda::device::get(j);
                auto access = cuda::device::peer_to_peer::can_access(device, peer);
                std::cout << std::setw(1) << ((access) ? 1 : 0);
            } else {
                std::cout << std::setw(1) << 1;
            }
        }
        std::cout << "\n";
    }
}

void list_devices(const cuda::device::id_t numGPUs)
{
    //output devices
    for (int i = 0; i < numGPUs; i++) {
        auto device = cuda::device::get(i);
        auto properties = device.properties();
        std::cout << "Device: " << device.id() << ", " << properties.name << ", PCI location " << properties.pci_id() << '\n';
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
        p2p_mechanism = SM;
    }
    return { test_p2p_read, p2p_mechanism };
}

int main(int argc, char **argv)
{
    const auto numGPUs = cuda::device::count();

    command_line_options opts = handle_command_line(argc, argv);
    std::cout << "[P2P (Peer-to-Peer) GPU Bandwidth Latency Test]\n";

    list_devices(numGPUs);

    checkP2Paccess(numGPUs);

    print_connectivity_matrix(numGPUs);

    enum : bool {
        do_measure_p2p = true,
        dont_measure_p2p = false
    };

    outputBandwidthMatrix(opts.mechanism, numGPUs, dont_measure_p2p, P2P_WRITE);
    outputBandwidthMatrix(opts.mechanism, numGPUs, do_measure_p2p, P2P_WRITE);
    if (opts.test_p2p_read)
    {
        outputBandwidthMatrix(opts.mechanism, numGPUs, do_measure_p2p, P2P_READ);
    }
    outputBidirectionalBandwidthMatrix(opts.mechanism, numGPUs, dont_measure_p2p);
    outputBidirectionalBandwidthMatrix(opts.mechanism, numGPUs, do_measure_p2p);

    outputLatencyMatrix(opts.mechanism, numGPUs, dont_measure_p2p, P2P_WRITE);
    outputLatencyMatrix(opts.mechanism, numGPUs, do_measure_p2p, P2P_WRITE);
    if (opts.test_p2p_read)
    {
        outputLatencyMatrix(opts.mechanism, numGPUs, do_measure_p2p, P2P_READ);
    }
    std::cout << "\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n";
}
