// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_fixture.hpp"
#include "tt_fabric/fabric_router.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

TEST_F(ControlPlaneFixture, TestTGRouter8MeshInit) {
  const std::filesystem::path tg_cluster_desc_path = std::filesystem::path(tt::llrt::OptionsG.get_root_dir()) / "tests/tt_metal/tt_fabric/common/tg_cluster_desc.yaml";
  std::unique_ptr<tt_ClusterDescriptor> tg_cluster_desc = tt_ClusterDescriptor::create_from_yaml(tg_cluster_desc_path.string());

  // Testing code, split the TG cluster into meshes of 2x4
  std::vector<uint32_t> mesh_sizes = {1,1,1,1,8,8,8,8};
  std::vector<std::vector<chip_id_t>> mesh_id_to_physical_chip_id;
  std::vector<std::pair<mesh_id_t,chip_id_t>> physical_chip_id_to_mesh_id;
  mesh_id_to_physical_chip_id.resize(mesh_sizes.size());
  physical_chip_id_to_mesh_id.resize(tg_cluster_desc->get_number_of_chips());
  for (uint32_t mesh_id = 0; mesh_id< mesh_sizes.size(); mesh_id++) {
      mesh_id_to_physical_chip_id[mesh_id].resize(mesh_sizes[mesh_id]);
  }

  for (auto &[chip_with_mmio, physical_id]: tg_cluster_desc->get_chips_with_mmio()) {
      mesh_id_to_physical_chip_id[chip_with_mmio][0] = chip_with_mmio;
      physical_chip_id_to_mesh_id[chip_with_mmio] = std::make_pair(chip_with_mmio, 0);
  }
  int i = 0, j=0, k=0, l=0;
  for (const auto& [chip_id, eth_coord]: tg_cluster_desc->get_chip_locations()) {
      if (tg_cluster_desc->get_board_type(chip_id) == BoardType::GALAXY) {
          switch (std::get<1>(eth_coord)) {
              case 7:
              case 6:
                  mesh_id_to_physical_chip_id[4][i] = chip_id;
                  physical_chip_id_to_mesh_id[chip_id] = std::make_pair(4, i);
                  i++;
                  break;
              case 5:
              case 4:
                  mesh_id_to_physical_chip_id[5][j] = chip_id;
                  physical_chip_id_to_mesh_id[chip_id] = std::make_pair(5, j);
                  j++;
                  break;
              case 3:
              case 2:
                  mesh_id_to_physical_chip_id[6][k] = chip_id;
                  physical_chip_id_to_mesh_id[chip_id] = std::make_pair(6, k);
                  k++;
                  break;
              case 1:
              case 0:
                  mesh_id_to_physical_chip_id[7][l] = chip_id;
                  physical_chip_id_to_mesh_id[chip_id] = std::make_pair(7, l);
                  l++;
                  break;
              default:
                  break;
        }

      }
  }

  // Create adjacency list
  for (auto mesh_id = 0; mesh_id < mesh_sizes.size(); mesh_id++) {
      std::cout << "Mesh " << mesh_id << " : ";
      for (auto chip_id: mesh_id_to_physical_chip_id[mesh_id]) {
          std::cout << chip_id << " ";
      }
      std::cout << std::endl;
  }

  // Create IntraMesh/InterMesh Connectivity
  std::vector<std::vector<std::unordered_map<chip_id_t, std::vector<port_id_t>>>> intra_mesh_connectivity;
  std::vector<std::vector<std::unordered_map<mesh_id_t, std::vector<port_id_t>>>> inter_mesh_connectivity;
  intra_mesh_connectivity.resize(mesh_sizes.size());
  inter_mesh_connectivity.resize(mesh_sizes.size());
  for (uint32_t mesh_id = 0; mesh_id< mesh_sizes.size(); mesh_id++) {
      intra_mesh_connectivity[mesh_id].resize(mesh_sizes[mesh_id]);
      inter_mesh_connectivity[mesh_id].resize(mesh_sizes[mesh_id]);
  }

  for (const auto& [physical_chip_id, connections]: tg_cluster_desc->get_ethernet_connections()) {
      auto [mesh_id, chip_id] = physical_chip_id_to_mesh_id[physical_chip_id];
      for (const auto& [port, connected_chip_port_pair]: connections) {
          auto &connected_physical_chip_id = std::get<0>(connected_chip_port_pair);
          auto &[connected_mesh_id, connected_chip_id] = physical_chip_id_to_mesh_id[connected_physical_chip_id];
          if (mesh_id == connected_mesh_id) {
              intra_mesh_connectivity[mesh_id][chip_id][connected_chip_id].push_back(port);
          } else {
              inter_mesh_connectivity[mesh_id][chip_id][connected_mesh_id].push_back(port);
          }
      }
  }
  for (uint32_t mesh_id = 0; mesh_id < intra_mesh_connectivity.size(); mesh_id++) {
      std::cout << "Mesh " << mesh_id << " : " << std::endl;
      for (uint32_t chip_id = 0; chip_id < intra_mesh_connectivity[mesh_id].size(); chip_id++) {
          std::cout << "   Chip " << chip_id << " : ";
          for (auto [connected_chip_id, ports]: intra_mesh_connectivity[mesh_id][chip_id]) {
              for (auto port: ports) {
                  std::cout << connected_chip_id << "(" << (uint32_t)port << ") ";
              }
          }
          std::cout << std::endl;
      }
  }
  for (uint32_t mesh_id = 0; mesh_id < inter_mesh_connectivity.size(); mesh_id++) {
      std::cout << "Mesh " << mesh_id << " : " << std::endl;
      for (uint32_t chip_id = 0; chip_id < intra_mesh_connectivity[mesh_id].size(); chip_id++) {
          std::cout << "   Chip " << chip_id << " : ";
          for (auto [connected_mesh_id, ports]: inter_mesh_connectivity[mesh_id][chip_id]) {
              for (auto port: ports) {
                  std::cout << connected_mesh_id << "(" << (uint32_t)port << ") ";
              }
          }
          std::cout << std::endl;
      }
  }
}

TEST_F(ControlPlaneFixture, TestTGRouterMeshInit) {
  const std::filesystem::path tg_cluster_desc_path = std::filesystem::path(tt::llrt::OptionsG.get_root_dir()) / "tests/tt_metal/tt_fabric/common/tg_cluster_desc.yaml";
  std::unique_ptr<tt_ClusterDescriptor> tg_cluster_desc = tt_ClusterDescriptor::create_from_yaml(tg_cluster_desc_path.string());

  tt::tt_fabric::FabricRouter fabric_router;
}
}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
