/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard launch configurations.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_MULTI_WRAPPERS_LAUNCH_CONFIGURATION_HPP
#define CUDA_API_WRAPPERS_MULTI_WRAPPERS_LAUNCH_CONFIGURATION_HPP

#include "../launch_configuration.hpp"
#include "../kernel.hpp"
#include "../device.hpp"
#include "../event.hpp"

namespace cuda {

namespace detail_ {

inline void validate_compatibility(
	const kernel_t& kernel,
	launch_configuration_t launch_config) noexcept(false)
{
	validate(launch_config);
	validate_block_dimension_compatibility(kernel, launch_config.dimensions.block);
	//  Uncomment if we actually get such checks
	//	validate_grid_dimension_compatibility(kernel, launch_config.dimensions.grid);
	validate_compatibility(kernel.device(), launch_config);
}

#if CUDA_VERSION >= 12000
inline CUlaunchConfig marshal(
	const launch_configuration_t &config,
	const stream::handle_t stream_handle,
	span<CUlaunchAttribute> attribute_storage) noexcept(true)
{
	unsigned int num_attributes = 0;
	// TODO: What about CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW ?
	if (config.block_cooperation) {
		auto &attr_value = attribute_storage[num_attributes++];
		attr_value.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
		attr_value.value.cooperative = 1;
	}
	if (grid::dimensions_t::point() != config.clustering.cluster_dimensions) {
		auto &attr_value = attribute_storage[num_attributes++];
		attr_value.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
		attr_value.value.clusterDim.x = config.clustering.cluster_dimensions.x;
		attr_value.value.clusterDim.y = config.clustering.cluster_dimensions.y;
		attr_value.value.clusterDim.z = config.clustering.cluster_dimensions.z;
	}
	if (config.clustering.scheduling_policy != cluster_scheduling_policy_t::default_) {
		auto &attribute = attribute_storage[num_attributes++];
		attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
		attribute.value.clusterSchedulingPolicyPreference =
			static_cast<CUclusterSchedulingPolicy>(config.clustering.scheduling_policy);
	}
	// TODO: CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
	if (config.programmatically_dependent_launch) {
		auto &attr_value = attribute_storage[num_attributes++];
		attr_value.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
		attr_value.value.programmaticStreamSerializationAllowed = 1;
	}
	if (config.programmatic_completion.event) {
		auto &attr_value = attribute_storage[num_attributes++];
		attr_value.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT;
		attr_value.value.programmaticEvent.event = config.programmatic_completion.event->handle();
		// TODO: What about the flags?
		attr_value.value.programmaticEvent.triggerAtBlockStart =
			config.programmatic_completion.trigger_event_at_block_start;
	}
	// What about CU_LAUNCH_ATTRIBUTE_PRIORITY ?
	if (config.in_remote_memory_synchronization_domain) {
		auto &attr_value = attribute_storage[num_attributes++];
		attr_value.id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
		attr_value.value.memSyncDomain = CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE;
	}
	// What about CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP ?
	attribute_storage[num_attributes] = {CU_LAUNCH_ATTRIBUTE_IGNORE, {}};

	return {
		config.dimensions.grid.x,
		config.dimensions.grid.y,
		config.dimensions.grid.z,
		config.dimensions.block.x,
		config.dimensions.block.y,
		config.dimensions.block.z,
		config.dynamic_shared_memory_size,
		stream_handle,
		attribute_storage.data(),
		num_attributes
	};
}
#endif // CUDA_VERSION >= 12000

} // namespace detail_

} // namespace cuda

#endif //CUDA_API_WRAPPERS_MULTI_WRAPPERS_LAUNCH_CONFIGURATION_HPP
