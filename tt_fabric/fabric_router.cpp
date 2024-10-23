// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_router.hpp"

namespace tt::tt_fabric {
FabricRouter::FabricRouter() {
  // TODO: temp hard code TG mesh setup, make GraphMesh and grab the info from there
  std::vector<uint32_t> mesh_sizes = {1,1,1,1,32};
  std::vector<std::vector<std::unordered_map<mesh_id_t, RouterEdge>>> inter_mesh_connectivity;
  std::vector<std::vector<std::unordered_map<mesh_id_t, RouterEdge>>> intra_mesh_connectivity;
  inter_mesh_connectivity.resize(mesh_sizes.size());
  intra_mesh_connectivity.resize(mesh_sizes.size());
  // Size the conncectivity vectors
  for (uint32_t mesh_id = 0; mesh_id< 5; mesh_id++) {
      inter_mesh_connectivity[mesh_id].resize(mesh_sizes[mesh_id]);
      intra_mesh_connectivity[mesh_id].resize(mesh_sizes[mesh_id]);
  }
  // Fill connectivity for mmio
  for (uint32_t mesh_id = 0; mesh_id< 4; mesh_id++) {
      intra_mesh_connectivity[mesh_id][0] = {};
      inter_mesh_connectivity[mesh_id][0] = {{4, RouterEdge{.weights = {0, 0}}}}; // 2 connections
  }
  for (int i = 0; i<32; i++) {
      // Fill in connectivity for Galaxy Mesh
      int row_size = 4;
      int N = i - row_size;
      int E = i - 1;
      int S = i + row_size;
      int W = i + 1;
      if (N >= 0) {
          intra_mesh_connectivity[4][i].insert({N, RouterEdge{.weights = {0, 0, 0, 0}}});
      }
      if (E >= 0 && (E / row_size == i / row_size)) {
          intra_mesh_connectivity[4][i].insert({E, RouterEdge{.weights = {0, 0, 0, 0}}});
      }
      if (S < 32) {
          intra_mesh_connectivity[4][i].insert({S, RouterEdge{.weights = {0, 0, 0, 0}}});
      }
      if (W < 32 && (W / row_size == i / row_size)) {
          intra_mesh_connectivity[4][i].insert({W, RouterEdge{.weights = {0, 0, 0, 0}}});
      }
  }
  inter_mesh_connectivity[4][0] = {{0, RouterEdge{.weights = {0}}}}; // 1 connection to Mesh 0
  inter_mesh_connectivity[4][4] = {{0, RouterEdge{.weights = {0}}}}; // 1 connection to Mesh 0
  inter_mesh_connectivity[4][8] = {{1, RouterEdge{.weights = {0}}}}; // 1 connection to Mesh 1
  inter_mesh_connectivity[4][12] = {{1, RouterEdge{.weights = {0}}}}; // 1 connection to Mesh 1
  inter_mesh_connectivity[4][16] = {{2, RouterEdge{.weights = {0}}}}; // 1 connection to Mesh 2
  inter_mesh_connectivity[4][20] = {{2, RouterEdge{.weights = {0}}}}; // 1 connection to Mesh 2
  inter_mesh_connectivity[4][24] = {{3, RouterEdge{.weights = {0}}}}; // 1 connection to Mesh 3
  inter_mesh_connectivity[4][28] = {{3, RouterEdge{.weights = {0}}}}; // 1 connection to Mesh 3

  // Print Connectivity
  for (uint32_t mesh_id = 0; mesh_id < intra_mesh_connectivity.size(); mesh_id++) {
      std::cout << "Mesh " << mesh_id << " : " << std::endl;
      for (uint32_t chip_id = 0; chip_id < intra_mesh_connectivity[mesh_id].size(); chip_id++) {
          std::cout << "   Chip " << chip_id << " : ";
          for (auto [connected_chip_id, edge]: intra_mesh_connectivity[mesh_id][chip_id]) {
              for (auto weight: edge.weights) {
                  std::cout << connected_chip_id << "(" << (uint32_t)weight << ") ";
              }
          }
          std::cout << std::endl;
      }
  }
  for (uint32_t mesh_id = 0; mesh_id < inter_mesh_connectivity.size(); mesh_id++) {
      std::cout << "Mesh " << mesh_id << " : " << std::endl;
      for (uint32_t chip_id = 0; chip_id < inter_mesh_connectivity[mesh_id].size(); chip_id++) {
          std::cout << "   Chip " << chip_id << " : ";
          for (auto [connected_chip_id, edge]: inter_mesh_connectivity[mesh_id][chip_id]) {
              for (auto weight: edge.weights) {
                  std::cout << connected_chip_id << "(" << (uint32_t)weight << ") ";
              }
          }
          std::cout << std::endl;
      }
  }


  // Initialize the fabric router
  this->intra_mesh_table_.resize(mesh_sizes.size());
  this->inter_mesh_table_.resize(mesh_sizes.size());
  for (mesh_id_t mesh_id = 0; mesh_id< mesh_sizes.size(); mesh_id++) {
    this->intra_mesh_table_[mesh_id].resize(mesh_sizes[mesh_id]);
    for (auto& devices_in_mesh: this->intra_mesh_table_[mesh_id]) {
          devices_in_mesh.resize(mesh_sizes[mesh_id]);
    }
    for (auto& devices_in_mesh: this->inter_mesh_table_[mesh_id]) {
          devices_in_mesh.resize(mesh_sizes.size());
    }
  }
}

} // namespace tt::tt_fabric
