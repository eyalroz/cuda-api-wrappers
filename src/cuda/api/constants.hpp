/**
 * @file constants.hpp
 *
 * @brief Fundamental CUDA-related constants and enumerations,
 * not dependent on any more complex abstractions, placed
 * in relevant namespaces.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_CONSTANTS_HPP_
#define CUDA_API_WRAPPERS_CONSTANTS_HPP_

#include <cuda/common/types.hpp>

namespace cuda {

/**
* CUDA's NVCC allows use the use of the warpSize identifier, without having
* to define it. Un(?)fortunately, warpSize is not a compile-time constant; it
* is replaced at some point with the appropriate immediate value which goes into,
* the SASS instruction as a literal. This is apparently due to the theoretical
* possibility of different warp sizes in the future. However, it is useful -
* both for host-side and more importantly for device-side code - to have the
* warp size available at compile time. This allows all sorts of useful
* optimizations, as well as its use in constexpr code.
*
* If nVIDIA comes out with 64-lanes-per-warp GPUs - we'll refactor this.
*/
enum : native_word_t { warp_size          = 32 };

namespace stream {

// Would have called it "default" but that's a reserved word;
// Would have liked to make this an enum, but pointers are
// not appropriate for that
/**
 * The CUDA runtime provides a default stream on which work
 * is scheduled when no stream is specified; for those API calls
 * where you need to specify the relevant stream's ID, and want to
 * specify the default, this is what you use.
 */
const stream::id_t default_stream_id = nullptr;

} // namespace stream

namespace device {

enum : device::id_t {
	/**
	 * If the CUDA runtime has not been set to a specific device, this
	 * is the ID of the device it defaults to.
	 */
	default_device_id = 0
};

} // namespace device

namespace event {

/**
 * Synchronization option for @ref cuda::event_t 's
 */
enum : bool {
	/**
	 * The thread calling event_.synchronize() will enter
	 * a busy-wait loop; this (might) minimize delay between
	 * kernel execution conclusion and control returning to
	 * the thread, but is very wasteful of CPU time.
	 */
	sync_by_busy_waiting = false,
	/**
	 * The thread calling event_.synchronize() will block -
	 * yield control of the CPU and will only become ready
	 * for execution after the kernel has completed its
	 * execution - at which point it would have to wait its
	 * turn among other threads. This does not waste CPU
	 * computing time, but results in a longer delay.
	 */
	sync_by_blocking = true,
};

/**
 * Should the CUDA Runtime API record timing information for
 * events as it schedules them?
 */
enum : bool {
	dont_record_timings = false,
	do_record_timings   = true,
};

/**
 * IPC usability option for {@ref cuda::event_t}'s
 */
enum : bool {
	not_interprocess = false,         //!< Can only be used by the process which created it
	interprocess = true,              //!< Can be shared between processes. Must not be able to record timings.
	single_process = not_interprocess
};

} // namespace event

/**
 * Thread block cooperativity control for kernel launches
 */
enum : bool {
	/** Thread groups may span multiple blocks, so that they can synchronize their actions */
	thread_blocks_may_cooperate = true,
	/** Thread blocks are not allowed to synchronize (the default, and likely faster, execution mode) */
	thread_blocks_may_not_cooperate = false
};

enum : memory::shared::size_t { no_dynamic_shared_memory = 0 };

enum : bool {
	do_take_ownership = true,
	do_not_take_ownership = false,
};

} // namespace cuda

#endif // CUDA_API_WRAPPERS_CONSTANTS_HPP_
