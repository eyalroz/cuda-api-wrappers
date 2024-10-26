#include <cuda/api.hpp>
#include "../common.hpp"

//
//__global__ void print_message(const char* message, size_t length)
//{
//	printf("%*s\n", length, message);
//}

static constexpr const size_t region_size { 1024UL }; // 64 MiB

void check_getters_and_rewrapping(const cuda::memory::pool_t &pool);

cuda::optional<cuda::device::id_t> maybe_get_p2p_peer_id(cuda::device::id_t device_id)
{
	const auto& dev = cuda::device::get(device_id);
	for(const auto& peer : cuda::devices()) {
		if (peer.id() == device_id) { continue; }
		if (cuda::device::peer_to_peer::can_access_each_other(dev, peer))  {
			return peer.id();
		}
	}
	return cuda::nullopt;
}

void play_with_attributes(const cuda::memory::pool_t &pool, const cuda::stream_t& stream)
{
#if CUDA_VERSION < 11300
	(void) stream;
#endif
	bool allow_internal_deps = pool.get_attribute<CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES>();
	bool allow_opportunistic = pool.get_attribute<CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC>();
	bool when_dependent_on_free = pool.get_attribute<CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES>();
	if (not (allow_internal_deps and allow_opportunistic and when_dependent_on_free)) {
		die_("Pool reuse policy differs from expect default");
	}
	auto reuse_policy = pool.reuse_policy();
	if (   reuse_policy.when_dependent_on_free != when_dependent_on_free
	    or reuse_policy.independent_but_actually_freed != allow_opportunistic
	    or reuse_policy.allow_waiting_for_frees != allow_internal_deps)
	{
		die_("Some aggregate reuse policy fields doesn't equal attributes obtained individually");
	}
	reuse_policy.when_dependent_on_free = false;

	pool.set_attribute<CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES>(false);
	pool.set_attribute<CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC>(false);
	pool.set_attribute<CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES>(false);

	allow_internal_deps = pool.get_attribute<CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES>();
	allow_opportunistic = pool.get_attribute<CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC>();
	when_dependent_on_free = pool.get_attribute<CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES>();

	if (allow_internal_deps) {
		die_("Pool reuse policy ALLOW_INTERNAL_DEPS is true despite having been set to false!");
	}

	if (allow_opportunistic) {
		die_("Pool reuse policy ALLOW_OPPORTUNISTIC is true despite having been set to false!");
	}

	if (when_dependent_on_free) {
		die_("Pool reuse policy FOLLOW_EVENT_DEPENDENCIES is true despite having been set to false!");
	}

#if CUDA_VERSION >= 11300
	auto max_use_since_reset = pool.get_attribute<CU_MEMPOOL_ATTR_USED_MEM_HIGH>();
	if (max_use_since_reset != region_size) {
		die_("Unexpected 'high-use watermark' property value for a memory pool");
	}

	auto temp_region = pool.allocate(stream, 2 * region_size);

	auto currently_used_mem = pool.get_attribute<CU_MEMPOOL_ATTR_USED_MEM_CURRENT>();
	if (currently_used_mem != 3 * region_size) {
		die_("Unexpected amount of currently-used memory reported by a memory pool");
	}

	stream.enqueue.free(temp_region);

	currently_used_mem = pool.get_attribute<CU_MEMPOOL_ATTR_USED_MEM_CURRENT>();
	if (currently_used_mem != region_size) {
		die_("Unexpected amount of currently-used memory reported by a memory pool");
	}

	max_use_since_reset = pool.get_attribute<CU_MEMPOOL_ATTR_USED_MEM_HIGH>();
	if (max_use_since_reset != 3 * region_size) {
		die_("Unexpected 'high-use watermark' property value for a memory pool");
	}

	pool.set_attribute<CU_MEMPOOL_ATTR_USED_MEM_HIGH>(0);

	max_use_since_reset = pool.get_attribute<CU_MEMPOOL_ATTR_USED_MEM_HIGH>();
	if (max_use_since_reset != region_size) {
		die_("Unexpected 'high-use watermark' property value for a memory pool, after a reset: Expected "
			 + std::to_string(region_size) + " but got " + std::to_string(max_use_since_reset));
	}
#endif // CUDA_VERSION >= 11030

	pool.set_attribute<CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES>(true);
	pool.set_attribute<CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC>(true);
	pool.set_attribute<CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES>(true);
}

void copy_through_pool_allocation(
	const cuda::stream_t& stream,
	cuda::memory::region_t pool_allocated_region)
{
	auto overwritten = "This will initialize the host buffer and should get overwritten.";
	auto host_buffer_uptr = std::unique_ptr<char[]>(new char[region_size]); // replace this with make_unique in C++14
	auto host_buffer = cuda::span<char>{host_buffer_uptr.get(), region_size};
	strcpy(host_buffer.data(), overwritten);
	(pool_allocated_region.size() == region_size) or die_("Unexpected allocation size");
	auto message = [](const char* msg) {
		return cuda::span<const char>{msg, strlen(msg) + 1};
	}("I get copied around");
	stream.enqueue.copy(pool_allocated_region, message);
	auto async_device_region = stream.enqueue.allocate(region_size);
	stream.enqueue.copy(async_device_region, pool_allocated_region);
	stream.enqueue.copy(host_buffer, pool_allocated_region);
	stream.synchronize();
	if (strncmp(message.data(), host_buffer.data(), message.size()) != 0) {
		auto len_of_overwritten = strlen(overwritten);
		host_buffer[len_of_overwritten] = '\0';
		std::ostringstream oss;
		oss << "Unexpected resulting buffer on the host side.\n"
			<< "Expected:            \"" << message.data() << "\"\n"
			<< "Actual (up to  " << len_of_overwritten << " chars): \"" << host_buffer.data() << "\"\n";
		die_(oss.str());
	}
}

cuda::memory::permissions_t
try_forbidding_same_device_access(int device_id, cuda::memory::pool_t &pool)
{
	cuda::memory::permissions_t permissions;
	permissions.read = true;
	permissions.write = false;
	bool got_expected_exception = false;
	try {
		pool.set_permissions(cuda::device::get(device_id), permissions);
	}
	catch(std::invalid_argument&) {
		got_expected_exception = true;
	}
	if (not got_expected_exception) {
		die_("Unexpected success in setting a device's access get_permissions to a pool using"
		"that device; it should have failed");
	}
	return permissions;
}

// Should only run when peer and device can access each other
void try_writing_to_pool_allocation_without_permission(
	const cuda::stream_t&                stream,
	cuda::memory::pool_t&                pool,
	cuda::memory::region_t&              pool_allocated_region,
	cuda::memory::permissions_t&  permissions,
	cuda::device_t&                      peer)
{
	permissions.read = false;
	permissions.write = false;
	pool.set_permissions(peer, permissions);
	std::string str{"hello world"};
	stream.synchronize();
	auto stream_on_peer = peer.create_stream(cuda::stream::async);
	bool got_expected_exception = false;
	try {
		stream_on_peer.enqueue.copy(pool_allocated_region, str);
		stream_on_peer.synchronize();
	}
	catch (cuda::runtime_error &ex) {
		if (ex.code() != cuda::status::invalid_value) {
			throw (ex);
		}
		got_expected_exception = true;
	}
	if (not got_expected_exception) {
		die_("Did not get the expected exception");
	}
}

// Should only run when peer and device can access each other
void try_reading_from_pool_allocation_without_permission(
	cuda::memory::pool_t&                pool,
	cuda::memory::region_t&              pool_allocated_region,
	cuda::memory::permissions_t&  permissions,
	cuda::device_t&                      peer)
{
	permissions.read = false;
	permissions.write = false;
	pool.set_permissions(peer, permissions);
	auto host_buffer_uptr = std::unique_ptr<char[]>(new char[region_size]); // replace this with make_unique in C++14
	auto host_buffer = cuda::span<char>{host_buffer_uptr.get(), region_size};
	std::fill_n(host_buffer.begin(), host_buffer.size()-1, 'a');
	host_buffer[host_buffer.size()-1] = '\0';
	auto stream_on_peer = peer.create_stream(cuda::stream::async);
	bool got_expected_exception = false;
	try {
		stream_on_peer.enqueue.copy(host_buffer, pool_allocated_region);
		stream_on_peer.synchronize();
	}
	catch (cuda::runtime_error &ex) {
		if (ex.code() != cuda::status::invalid_value) {
			throw (ex);
		}
		got_expected_exception = true;
	}
	if (not got_expected_exception) {
		die_("Did not get the expected exception");
	}
}

int main(int argc, char** argv)
{
	auto device_id = choose_device(argc, argv);
	auto device = cuda::device::get(device_id);
	auto stream = device.create_stream(cuda::stream::async);
	auto pool = device.create_memory_pool();

	check_getters_and_rewrapping(pool);
	auto pool_allocated_region = pool.allocate(stream, region_size);
	play_with_attributes(pool, stream);
	copy_through_pool_allocation(stream, pool_allocated_region);

	cuda::memory::permissions_t permissions = try_forbidding_same_device_access(device_id, pool);

	auto maybe_peer_id = maybe_get_p2p_peer_id(device_id);
	if (maybe_peer_id) {
		auto peer = cuda::device::get(maybe_peer_id.value());

		try_writing_to_pool_allocation_without_permission(
			stream, pool, pool_allocated_region, permissions, peer);
		try_reading_from_pool_allocation_without_permission(
			pool, pool_allocated_region, permissions, peer);
	}
	else {
		std::cout << "No device can server as a peer of our chosen device, with P2P access to its memory;\n"
					 "Skipping P2P-related tests.";
	}

	stream.enqueue.free(pool_allocated_region);

	std::cout << "\nSUCCESS\n\n";
}

void check_getters_and_rewrapping(const cuda::memory::pool_t &pool)
{
	auto device_id = pool.device_id();
	auto handle = pool.handle();
	if (not pool.is_owning()) {
		die_("Memory pool proxy should be owning, but isn't");
	}

	static const bool is_not_owning { false };
	auto wrapped = cuda::memory::pool::wrap(device_id, handle, is_not_owning);
	if (wrapped != pool) {
		die_("Rewrapped pool proxy different than the pool proxy we queried");
	}
}
