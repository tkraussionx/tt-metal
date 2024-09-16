

std::vector<WorkerAttributes> build_worker_attributes(
    ttnn::ccl::RingTopology const& topology_config,
    std::vector<CoreCoord> const& worker_cores_list,
    std::optional<std::vector<CoreCoord>> const& second_worker_cores_list,

    std::size_t num_links,
    std::size_t num_channels_per_link,
    std::function<bool(std::size_t)> is_buffer_in_clockwise_direction_fn);
