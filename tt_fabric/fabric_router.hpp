// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "tt_metal/llrt/tt_cluster.hpp"


namespace tt::tt_fabric {

using port_id_t = uint32_t;
using mesh_id_t = uint32_t;
using RoutingTable = std::vector<std::vector<std::vector<port_id_t>>>; // [mesh_id][chip_id][target_chip_or_mesh_id]

enum class RoutingDirection	 {
	NORTH	= 0,
	EAST = 1,
	SOUTH = 2,
	WEST = 3,
	CENTRAL = 4,
};

struct RouterEdge {
  std::vector<std::uint32_t> weights;
};

class FabricConnectivity {
  //  std::vector<RouterNode*>;
};

class FabricRouter {

public:

   FabricRouter();
  ~FabricRouter() = default;

	void dump_to_yaml();
	void load_from_yaml();

	RoutingTable get_intra_mesh_table();
	RoutingTable get_inter_mesh_table();

private:
	// configurable in future architectures
	const uint32_t max_nodes_in_mesh_ = 1024;
  const uint32_t max_num_meshes_ = 1024;

  std::vector<uint32_t> mesh_sizes;

	RoutingTable intra_mesh_table_;
  RoutingTable inter_mesh_table_;

  // Dimension Ordered Routing
	void generate_routing_table(chip_id_t chip_id);

};

} // namespace tt::tt_fabric
