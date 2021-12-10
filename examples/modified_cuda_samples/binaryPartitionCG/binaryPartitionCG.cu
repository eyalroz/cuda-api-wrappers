/**
 * Derived from the nVIDIA CUDA 11.4 samples by
 *
 *   Eyal Rozenberg <eyalroz1@gmx.com>
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 */

/*
 * Minimum SM 3.5 ... check that.
 *
 * This sample illustrates basic usage of binary partition cooperative groups
 * within the thread block tile when divergent path exists.
 * 1.) Each thread loads a value from random array.
 * 2.) then checks if it is odd or even.
 * 3.) create binary partition group based on the above predicate
 * 4.) we count the number of odd/even in the group based on size of the binary groups
 * 5.) write it global counter of odd.
 * 6.) sum the values loaded by individual threads(using reduce) and write it to global 
 *     even & odd elements sum.
 *
 * **NOTE** : binary_partition results in splitting warp into divergent thread groups
              this is not good from performance perspective, but in cases where warp 
              divergence is inevitable one can use binary_partition group.
*/

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../helper_cuda.hpp"
#include <cuda/runtime_api.hpp>

namespace cg = cooperative_groups;

void initOddEvenArr(int *inputArr, unsigned int size)
{
    for (unsigned int i=0; i < size; i++)
    {
        inputArr[i] = rand() % 50;
    }
}


/**
 * CUDA kernel device code
 * 
 * Creates cooperative groups and performs odd/even counting & summation.
 */
__global__ void oddEvenCountAndSumCG(int *inputArr, int *numOfOdds, int *sumOfOddAndEvens, unsigned int size)
{
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    for (auto i = grid.thread_rank(); i < size; i += grid.size())
    {
        int elem = inputArr[i];
        auto subTile = cg::binary_partition(tile32, elem & 1);
        if (elem & 1) // Odd numbers group
        {
            int oddGroupSum = cg::reduce(subTile, elem, cg::plus<int>());

            if (subTile.thread_rank() == 0)
            {
                // Add number of odds present in this group of Odds.
                atomicAdd(numOfOdds, (int) subTile.size());

                // Add local reduction of odds present in this group of Odds.
                atomicAdd(&sumOfOddAndEvens[0], oddGroupSum);

            }
        }
        else // Even numbers group
        {
            int evenGroupSum = cg::reduce(subTile, elem, cg::plus<int>());

            if (subTile.thread_rank() == 0)
            {
                // Add local reduction of even present in this group of evens.
                atomicAdd(&sumOfOddAndEvens[1], evenGroupSum);
            }
        }
        // reconverge warp so for next loop iteration we ensure convergence of 
        // above diverged threads to perform coalesced loads of inputArr.
        cg::sync(tile32);
    }
}


/**
 * Host main routine
 */
int main(int argc, const char **argv)
{
    auto device = chooseCudaDevice(argc, argv);

    unsigned int arrSize = 1024 * 100;

	auto h_inputArr = cuda::memory::host::make_unique<int[]>(arrSize);
	auto h_numOfOdds = cuda::memory::host::make_unique<int>();
	auto h_sumOfOddEvenElems = cuda::memory::host::make_unique<int[]>(2);
    initOddEvenArr(h_inputArr.get(), arrSize);
   
    auto stream = device.create_stream(cuda::stream::async);
	// Note: With CUDA 11, we could allocate these asynchronously on the stream
	auto d_inputArr = cuda::memory::device::make_unique<int[]>(device, arrSize);
	auto d_numOfOdds = cuda::memory::device::make_unique<int>(device);
	auto d_sumOfOddEvenElems = cuda::memory::device::make_unique<int[]>(device, 2);

	// Note: There's some code repetition here; unique pointers don't also keep track of the allocated size.
	// Unfortunately, the standard library does not offer an owning dynamically-allocated memory region
	// abstraction, other than std::vector which is not CUDA-device-friendly
	stream.enqueue.copy(d_inputArr.get(), h_inputArr.get(), sizeof(int)*arrSize);
	stream.enqueue.memzero(d_numOfOdds.get(), sizeof(int));
	stream.enqueue.memzero(d_sumOfOddEvenElems.get(), sizeof(int)*2);

	auto kernel = cuda::kernel::wrap(device, oddEvenCountAndSumCG);
	auto dims = kernel.min_grid_params_for_max_occupancy();
	auto launch_config = cuda::make_launch_config(dims);
		// Note: While the kernel uses the "cooperative groups" CUDA-C++ headers,
		// it doesn't involve any inter-block cooperation, so we don't indicate
		// block cooperation in the launch configuration

	if (dims.block.dimensionality() != 1 or dims.grid.dimensionality() != 1) {
		throw std::logic_error("Unexpected grid parameters received from kernel_t::min_grid_params_for_max_occupancy - "
			"block dims have " + std::to_string(dims.block.dimensionality()) + " dimensions and "
			"grid dims have " + std::to_string(dims.grid.dimensionality()) + " dimensions");
	}
	std::cout << "\nLaunching " << dims.block.volume() << " blocks with " << dims.grid.volume() << " threads...\n\n";

	stream.enqueue.kernel_launch(kernel, launch_config, d_inputArr.get(), d_numOfOdds.get(), d_sumOfOddEvenElems.get(), arrSize);

	cuda::memory::async::copy(h_numOfOdds.get(), d_numOfOdds.get(), sizeof(int), stream);
	cuda::memory::async::copy(h_sumOfOddEvenElems.get(), d_sumOfOddEvenElems.get(), sizeof(int)*2, stream);

	stream.synchronize();

    std::cout
		<< "Array size = " << arrSize
		<< " Num of Odds = " << h_numOfOdds.get()[0]
		<< " Sum of Odds = " << h_sumOfOddEvenElems.get()[0]
		<< " Sum of Evens " << h_sumOfOddEvenElems.get()[1]
		<< "\n";

    std::cout << "\n\nSUCCESS\n\n"; // Actually, we don't even check the sum, but... that's what NVIDIA wrote.

    return EXIT_SUCCESS;
}
