/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2023, Eyal Rozenberg <eyalroz1@gmx.com>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// System includes
#include <assert.h>
#include <stdio.h>

#include <climits>
#include <vector>

// CUDA runtime
#include <cuda/api.hpp>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#define THREADS_PER_BLOCK 512
#define ALLOWABLE_VARIANCE 1.e-6f
#define NUM_ELEMENTS 8000000

// Stores the square of each input element in output array
__global__ void squareArray(const float *input, float *output,
							int numElements)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numElements) {
		output[idx] = input[idx] * input[idx];
	}
}

// Stores the negative of each input element in output array
__global__ void negateArray(const float *input, float *output,
							int numElements)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numElements) {
		output[idx] = input[idx] * -1;
	}
}

struct negSquareArrays {
	float *input;
	float *square;
	float *negSquare;
	int numElements;
	size_t bytes;
	size_t numBlocks;
};

void fillRandomly(float *array, int numElements)
{
	for (int n = 0; n < numElements; n++) {
		array[n] = rand() / (float) RAND_MAX;
	}
}

void resetOutputArrays(negSquareArrays *hostArrays)
{
	fillRandomly(hostArrays->square, hostArrays->numElements);
	fillRandomly(hostArrays->negSquare, hostArrays->numElements);
}

void prepareHostArrays(negSquareArrays *hostArrays)
{
	hostArrays->numElements = NUM_ELEMENTS;
	size_t bytes = hostArrays->numElements * sizeof(float);

	size_t numBlocks = hostArrays->numElements / (size_t) THREADS_PER_BLOCK;
	if ((numBlocks % (size_t) THREADS_PER_BLOCK) != 0) {
		numBlocks++;
	}

	hostArrays->input = (float *) malloc(bytes);
	hostArrays->square = (float *) malloc(bytes);
	hostArrays->negSquare = (float *) malloc(bytes);
	hostArrays->bytes = bytes;
	hostArrays->numBlocks = numBlocks;

	fillRandomly(hostArrays->input, hostArrays->numElements);
	fillRandomly(hostArrays->square, hostArrays->numElements);
	fillRandomly(hostArrays->negSquare, hostArrays->numElements);
}

cuda::graph::instance_t createFreeGraph(float *dPtr)
{
	cudaGraphNode_t freeNode;

	auto graph = cuda::graph::create();
	auto node = graph.insert.node<cuda::graph::node::kind_t::memory_free>(dPtr);
	return graph.instantiate();
}

/**
 * Demonstrates explicitly creating a CUDA graph including memory nodes.
 * createNegateSquaresGraphWithStreamCapture constructs an equivalent graph
 * using stream capture.
 *
 * If d_negSquare_out is non null, then:
 * 1) d_negSquare will not be freed;
 * 2) the value of d_negSquare_out will be set to d_negSquare.
 *
 * Diagram of the graph constructed by createNegateSquaresGraphExplicitly:
 *
 * alloc d_input
 *       |
 * alloc d_square
 *       |
 * Memcpy a to device
 *       |
 * launch kernel squareArray ------->---- Memcpy d_square to host
 *       |                                      |
 * free d_input                                 |
 *       |                                      |
 * allocate d_negSquare                         |
 *       |                                      |
 * launch kernel negateArray -------->--- free d_square
 *       |
 * Memcpy d_negSquare to host
 *       |
 * free d_negSquare
 */
std::pair<cuda::graph::instance_t, float*>
createNegateSquaresGraphExplicitly(int device, negSquareArrays *hostArrays, bool do_neg_squares)
{
	// Array buffers on device
	float *d_input, *d_square, *d_negSquare;

	// Memory allocation parameters
	cudaMemAllocNodeParams allocParams;
	memset(&allocParams, 0, sizeof(allocParams));
	allocParams.bytesize = hostArrays->bytes;
	allocParams.poolProps.allocType = cudaMemAllocationTypePinned;
	allocParams.poolProps.location.id = device;
	allocParams.poolProps.location.type = cudaMemLocationTypeDevice;

	// Kernel launch parameters
	cudaKernelNodeParams kernelNodeParams = {0};
	kernelNodeParams.gridDim = dim3(hostArrays->numBlocks, 1, 1);
	kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
	kernelNodeParams.sharedMemBytes = 0;
	kernelNodeParams.extra = NULL;

	cudaGraph_t graph;
	cudaGraphNode_t allocNodeInput, allocNodeSquare, allocNodeNegSquare;
	cudaGraphNode_t copyNodeInput, copyNodeSquare, copyNodeNegSquare;
	cudaGraphNode_t squareKernelNode, negateKernelNode;
	cudaGraphNode_t freeNodeInput, freeNodeSquare;

	// Buffer for storing graph node dependencies
	std::vector<cudaGraphNode_t> nodeDependencies;

	checkCudaErrors(cudaGraphCreate(&graph, 0));

	checkCudaErrors(
		cudaGraphAddMemAllocNode(&allocNodeInput, graph, NULL, 0, &allocParams));
	d_input = (float *) allocParams.dptr;

	// To keep the graph structure simple (fewer branching dependencies),
	// allocNodeSquare should depend on allocNodeInput
	checkCudaErrors(cudaGraphAddMemAllocNode(&allocNodeSquare, graph,
		&allocNodeInput, 1, &allocParams));
	d_square = (float *) allocParams.dptr;

	// copyNodeInput needs to depend on allocNodeInput because copyNodeInput
	// writes to d_input. It does so here indirectly through allocNodeSquare.
	checkCudaErrors(cudaGraphAddMemcpyNode1D(
		&copyNodeInput, graph, &allocNodeSquare, 1, d_input, hostArrays->input,
		hostArrays->bytes, cudaMemcpyHostToDevice));

	void *squareKernelArgs[3] = {(void *) &d_input, (void *) &d_square,
								 (void *) &(hostArrays->numElements)};
	kernelNodeParams.func = (void *) squareArray;
	kernelNodeParams.kernelParams = (void **) squareKernelArgs;

	// Square kernel depends on copyNodeInput to ensure all data is on the device
	// before kernel launch.
	checkCudaErrors(cudaGraphAddKernelNode(&squareKernelNode, graph,
		&copyNodeInput, 1, &kernelNodeParams));

	checkCudaErrors(cudaGraphAddMemcpyNode1D(
		&copyNodeSquare, graph, &squareKernelNode, 1, hostArrays->square,
		d_square, hostArrays->bytes, cudaMemcpyDeviceToHost));

	// Free of d_input depends on the square kernel to ensure that d_input is not
	// freed while being read by the kernel. It also depends on the alloc of
	// d_input via squareKernelNode > copyNodeInput > allocNodeSquare >
	// allocNodeInput.
	checkCudaErrors(cudaGraphAddMemFreeNode(&freeNodeInput, graph,
		&squareKernelNode, 1, d_input));

	// Allocation of C depends on free of A so CUDA can reuse the virtual address.
	checkCudaErrors(cudaGraphAddMemAllocNode(&allocNodeNegSquare, graph,
		&freeNodeInput, 1, &allocParams));
	d_negSquare = (float *) allocParams.dptr;

	if (d_negSquare == d_input) {
		printf(
			"Check verified that d_negSquare and d_input share a virtual "
			"address.\n");
	}

	void *negateKernelArgs[3] = {(void *) &d_square, (void *) &d_negSquare,
								 (void *) &(hostArrays->numElements)};
	kernelNodeParams.func = (void *) negateArray;
	kernelNodeParams.kernelParams = (void **) negateKernelArgs;

	checkCudaErrors(cudaGraphAddKernelNode(
		&negateKernelNode, graph, &allocNodeNegSquare, 1, &kernelNodeParams));

	nodeDependencies.push_back(copyNodeSquare);
	nodeDependencies.push_back(negateKernelNode);
	checkCudaErrors(cudaGraphAddMemFreeNode(&freeNodeSquare, graph,
		nodeDependencies.data(),
		nodeDependencies.size(), d_square));
	nodeDependencies.clear();

	checkCudaErrors(cudaGraphAddMemcpyNode1D(
		&copyNodeNegSquare, graph, &negateKernelNode, 1, hostArrays->negSquare,
		d_negSquare, hostArrays->bytes, cudaMemcpyDeviceToHost));

	if (d_negSquare_out == NULL) {
		cudaGraphNode_t freeNodeNegSquare;
		checkCudaErrors(cudaGraphAddMemFreeNode(
			&freeNodeNegSquare, graph, &copyNodeNegSquare, 1, d_negSquare));
	}
	else {
		*d_negSquare_out = d_negSquare;
	}

	checkCudaErrors(cudaGraphInstantiate(graphExec, graph, NULL, NULL, 0));
	checkCudaErrors(cudaGraphDestroy(graph));
}

cuda::graph::instance_t
createNegateSquaresGraphExplicitly(int device, negSquareArrays *hostArrays)
{
	static constexpr const auto do_neg_squares { true };
	return createNegateSquaresGraphExplicitly(device, hostArrays, do_neg_squares).first;
}

/**
 * Adds work to a CUDA stream which negates the square of values in the input
 * array.
 *
 * If d_negSquare_out is non null, then:
 * 1) d_negSquare will not be freed;
 * 2) the value of d_negSquare_out will be set to d_negSquare.
 *
 * Diagram of the stream operations in doNegateSquaresInStream
 * ---------------------------------------------------------------------
 * | STREAM                             | STREAM2                      |
 * ---------------------------------------------------------------------
 *
 * alloc d_input
 *       |
 * alloc d_square
 *       |
 * Memcpy a to device
 *       |
 * launch kernel squareArray
 *       |
 * record squareKernelCompleteEvent -->-- wait squareKernelCompleteEvent
 *       |                                      |
 * free d_input                                 |
 *       |                                      |
 * allocate d_negSquare                   Memcpy d_square to host
 *       |                                      |
 * launch kernel negateArray                    |
 *       |                                      |
 * record negateKernelCompleteEvent -->-- wait negateKernelCompleteEvent
 *       |                                      |
 * Memcpy d_negSquare to host                   |
 *       |                                free d_square
 * free d_negSquare                             |
 *       |                                      |
 * wait squareFreeEvent --------------<---- record squareFreeEvent
 */
void doNegateSquaresInStream(cudaStream_t stream1, negSquareArrays *hostArrays,
							 float **d_negSquare_out = NULL)
{
	float *d_input, *d_square, *d_negSquare;
	cudaStream_t stream2;
	cudaEvent_t squareKernelCompleteEvent, negateKernelCompleteEvent,
		squareFreeEvent;

	checkCudaErrors(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));

	checkCudaErrors(cudaEventCreate(&squareKernelCompleteEvent));
	checkCudaErrors(cudaEventCreate(&negateKernelCompleteEvent));
	checkCudaErrors(cudaEventCreate(&squareFreeEvent));

	// Virtual addresses are assigned synchronously when cudaMallocAsync is
	// called, thus there is no performace benefit gained by separating the
	// allocations into two streams.
	checkCudaErrors(cudaMallocAsync(&d_input, hostArrays->bytes, stream1));
	checkCudaErrors(cudaMallocAsync(&d_square, hostArrays->bytes, stream1));

	checkCudaErrors(cudaMemcpyAsync(d_input, hostArrays->input, hostArrays->bytes,
		cudaMemcpyHostToDevice, stream1));
	squareArray<<<hostArrays->numBlocks, THREADS_PER_BLOCK, 0, stream1>>>(
		d_input, d_square, hostArrays->numElements);
	checkCudaErrors(cudaEventRecord(squareKernelCompleteEvent, stream1));

	checkCudaErrors(cudaStreamWaitEvent(stream2, squareKernelCompleteEvent, 0));
	checkCudaErrors(cudaMemcpyAsync(hostArrays->square, d_square,
		hostArrays->bytes, cudaMemcpyDeviceToHost,
		stream2));

	checkCudaErrors(cudaFreeAsync(d_input, stream1));
	checkCudaErrors(cudaMallocAsync(&d_negSquare, hostArrays->bytes, stream1));
	negateArray<<<hostArrays->numBlocks, THREADS_PER_BLOCK, 0, stream1>>>(
		d_square, d_negSquare, hostArrays->numElements);
	checkCudaErrors(cudaEventRecord(negateKernelCompleteEvent, stream1));
	checkCudaErrors(cudaMemcpyAsync(hostArrays->negSquare, d_negSquare,
		hostArrays->bytes, cudaMemcpyDeviceToHost,
		stream1));
	if (d_negSquare_out == NULL) {
		checkCudaErrors(cudaFreeAsync(d_negSquare, stream1));
	}
	else {
		*d_negSquare_out = d_negSquare;
	}

	checkCudaErrors(cudaStreamWaitEvent(stream2, negateKernelCompleteEvent, 0));
	checkCudaErrors(cudaFreeAsync(d_square, stream2));
	checkCudaErrors(cudaEventRecord(squareFreeEvent, stream2));

	checkCudaErrors(cudaStreamWaitEvent(stream1, squareFreeEvent, 0));

	checkCudaErrors(cudaStreamDestroy(stream2));
	checkCudaErrors(cudaEventDestroy(squareKernelCompleteEvent));
	checkCudaErrors(cudaEventDestroy(negateKernelCompleteEvent));
	checkCudaErrors(cudaEventDestroy(squareFreeEvent));
}

/**
 * Demonstrates creating a CUDA graph including memory nodes using stream
 * capture. createNegateSquaresGraphExplicitly constructs an equivalent graph
 * without stream capture.
 */
cuda::graph::instance_t createNegateSquaresGraphWithStreamCapture(negSquareArrays *hostArrays,
											   float **d_negSquare_out = NULL)
{
	cudaGraph_t graph;
	cudaStream_t stream;

	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
	doNegateSquaresInStream(stream, hostArrays, d_negSquare_out);
	checkCudaErrors(cudaStreamEndCapture(stream, &graph));

	checkCudaErrors(cudaGraphInstantiate(graphExec, graph, NULL, NULL, 0));
	checkCudaErrors(cudaStreamDestroy(stream));
	checkCudaErrors(cudaGraphDestroy(graph));
}

void prepareRefArrays(negSquareArrays *hostArrays,
					  negSquareArrays *deviceRefArrays,
					  bool **foundValidationFailure)
{
	deviceRefArrays->bytes = hostArrays->bytes;
	deviceRefArrays->numElements = hostArrays->numElements;

	for (int i = 0; i < hostArrays->numElements; i++) {
		hostArrays->square[i] = hostArrays->input[i] * hostArrays->input[i];
		hostArrays->negSquare[i] = hostArrays->square[i] * -1;
	}

	checkCudaErrors(
		cudaMalloc((void **) &deviceRefArrays->negSquare, deviceRefArrays->bytes));
	checkCudaErrors(cudaMemcpy(deviceRefArrays->negSquare, hostArrays->negSquare,
		hostArrays->bytes, cudaMemcpyHostToDevice));

	checkCudaErrors(
		cudaMallocManaged((void **) foundValidationFailure, sizeof(bool)));
}

int checkValidationFailure(bool *foundValidationFailure)
{
	if (*foundValidationFailure) {
		std::cout << "Validation FAILURE!\n\n";
		*foundValidationFailure = false;
		return EXIT_FAILURE;
	}
	else {
		std::cout << "Validation PASSED!\n\n";
		return EXIT_SUCCESS;
	}
}

__global__ void validateGPU(float *d_negSquare, negSquareArrays devRefArrays,
							bool *foundValidationFailure)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float ref, diff;

	if (idx < devRefArrays.numElements) {
		ref = devRefArrays.negSquare[idx];
		diff = d_negSquare[idx] - ref;
		diff *= diff;
		ref *= ref;
		if (diff / ref > ALLOWABLE_VARIANCE) {
			*foundValidationFailure = true;
		}
	}
}

void validateHost(negSquareArrays *hostArrays, bool *foundValidationFailure)
{
	float ref, diff;

	for (int i = 0; i < hostArrays->numElements; i++) {
		ref = hostArrays->input[i] * hostArrays->input[i] * -1;
		diff = hostArrays->negSquare[i] - ref;
		diff *= diff;
		ref *= ref;
		if (diff / ref > ALLOWABLE_VARIANCE) {
			*foundValidationFailure = true;
		}
	}
}

int main(int argc, char **argv)
{
	negSquareArrays hostArrays, deviceRefArrays;

	auto launch_config = cuda::launch_config_builder()
		.block_dimensions(THREADS_PER_BLOCK)
		.grid_dimensions(hostArrays.numBlocks)
		.no_dynamic_shared_memory()
		.build();

	// Declare pointers for GPU buffers
	float *d_negSquare = NULL;
	bool *foundValidationFailure = NULL;

	srand(time(0));

	// Being very cavalier about our command-line arguments here...
	cuda::device::id_t device_id = (argc > 1) ? std::stoi(argv[1]) : cuda::device::default_device_id;
	auto device = cuda::device::get(device_id);

	if (cuda::version_numbers::driver() < cuda::version_numbers::make(11040)) {
		std::cout << "Waiving execution as driver does not support Graph Memory Nodes\n";
		exit(EXIT_SUCCESS);
	}

	if (not device.supports_memory_pools()) {
		std::cout << "Waiving execution as device does not support Memory Pools\n";
		exit(EXIT_SUCCESS);
	}
	std::cout << "Setting up sample.\n";

	prepareHostArrays(&hostArrays);
	prepareRefArrays(&hostArrays, &deviceRefArrays, &foundValidationFailure);
	auto stream = device.create_stream(cuda::stream::async);
	std::cout << "Setup complete.\n\n";

	std::cout << "Running negateSquares in a stream.\n";
	doNegateSquaresInStream(stream.handle(), &hostArrays);
	std::cout << "Validating negateSquares in a stream...\n";
	validateHost(&hostArrays, foundValidationFailure);
	checkValidationFailure(foundValidationFailure);
	resetOutputArrays(&hostArrays);

	{
		std::cout << "Running negateSquares in a stream-captured graph.\n";
		auto executable_graph_instance = createNegateSquaresGraphWithStreamCapture(&hostArrays);
		cuda::graph::launch(executable_graph_instance, stream);
		stream.synchronize();
		std::cout << "Validating negateSquares in a stream-captured graph...\n";
		validateHost(&hostArrays, foundValidationFailure);
		checkValidationFailure(foundValidationFailure);
		resetOutputArrays(&hostArrays);
	}

	{
		std::cout << "Running negateSquares in an explicitly constructed graph.\n";
		auto executable_graph_instance = createNegateSquaresGraphExplicitly(device.id(), &hostArrays);
		cuda::graph::launch(executable_graph_instance, stream);
		stream.synchronize();
		std::cout << "Validating negateSquares in an explicitly constructed graph...\n";
		validateHost(&hostArrays, foundValidationFailure);
		checkValidationFailure(foundValidationFailure);
		resetOutputArrays(&hostArrays);
	}
	// Each of the three examples below free d_negSquare outside the graph. As
	// demonstrated by validateGPU, d_negSquare can be accessed by outside the
	// graph before d_negSquare is freed.

	{
		std::cout << "Running negateSquares with d_negSquare freed outside the stream.\n";
		static constexpr const auto compute_neg_squares { true };
		auto pair = createNegateSquaresGraphExplicitly(device.id(), &hostArrays, compute_neg_squares);
		auto executable_graph_instance = std::move(pair.first);
		auto d_negSquare = std::move(pair.second);
		auto free_graph_instance = createFreeGraph(d_negSquare);
		cuda::graph::launch(executable_graph_instance, stream);
		stream.enqueue.kernel_launch(validateGPU, launch_config, d_negSquare, deviceRefArrays, foundValidationFailure);
		stream.synchronize();
		printf(
			"Validating negateSquares with d_negSquare freed outside the "
			"stream...\n");
		validateHost(&hostArrays, foundValidationFailure);
		checkValidationFailure(foundValidationFailure);
		resetOutputArrays(&hostArrays);

		std::cout << "Running negateSquares with d_negSquare freed outside the graph.\n";
		cuda::graph::launch(executable_graph_instance, stream);
		stream.enqueue.kernel_launch(validateGPU, launch_config, d_negSquare, deviceRefArrays, foundValidationFailure);
		stream.synchronize();
		printf(
			"Validating negateSquares with d_negSquare freed outside the graph...\n");
		checkValidationFailure(foundValidationFailure);
		resetOutputArrays(&hostArrays);
		// TODO: What about the instance vs the FreeC?
		printf(
			"Running negateSquares with d_negSquare freed in a different graph.\n");
		cuda::graph::launch(executable_graph_instance, stream);
		stream.enqueue.kernel_launch(validateGPU, launch_config, d_negSquare, deviceRefArrays, foundValidationFailure);
		cuda::graph::launch(free_graph_instance, stream);
		stream.synchronize();
		printf(
			"Validating negateSquares with d_negSquare freed in a different "
			"graph...\n");
		checkValidationFailure(foundValidationFailure);

	}

	std::cout << "\nSUCCESS\n";
}