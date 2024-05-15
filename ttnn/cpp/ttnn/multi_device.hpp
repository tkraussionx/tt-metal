// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "tt_eager/tensor/tensor.hpp"
#include "tt_metal/impl/device/multi_device.hpp"
#include "tt_metal/impl/trace/trace.hpp"
using Device = ttnn::Device;


namespace ttnn {

namespace multi_device {

using DeviceGrid = std::pair<int, int>;
using DeviceIds = std::vector<int>;


inline DeviceMesh open_device_mesh(const DeviceGrid& device_grid, const DeviceIds& device_ids, size_t l1_small_size) {
    return DeviceMesh(device_grid, device_ids, l1_small_size);
}

inline void close_device_mesh(DeviceMesh &multi_device) {
    multi_device.close_devices();
}

inline uint32_t begin_trace_capture(DeviceMesh* device, const uint32_t trace_buff_size, const uint8_t cq_id = 0) {
    auto workers = device->get_devices();
    uint32_t tid = Trace::next_id();
    for (auto& worker : workers) {
        worker->push_work(
            [worker, trace_buff_size, cq_id, tid] () mutable {
                worker->begin_trace(cq_id, tid, trace_buff_size);
            });
    }
    return tid;
}

void end_trace_capture(DeviceMesh* device, const uint32_t tid, const uint8_t cq_id = 0) {
    auto workers = device->get_devices();
    for (auto& worker : workers) {
        worker->push_work(
            [worker, cq_id, tid] () mutable {
                worker->end_trace(cq_id, tid);
            });
    }
}

inline void execute_trace(DeviceMesh* device, const uint32_t tid, const uint8_t cq_id = 0, bool blocking = true) {
    auto workers = device->get_devices();
    // If blocking, ensure that each worker thread blocks until device-local trace is completed
    for (auto& worker : workers) {
        worker->push_work(
            [worker, cq_id, tid, blocking] () mutable {
                worker->replay_trace(cq_id, tid, blocking);
            });
    }
    // If blocking, wait until worker threads have completed
    if (blocking) {
        for (auto& worker : workers) {
            worker->synchronize();
        }
    }
}

inline void release_trace(DeviceMesh* device, const uint32_t tid, const uint8_t cq_id) {
    auto workers = device->get_devices();
    for (auto& worker : workers) {
        worker->push_work(
            [worker, cq_id, tid] () mutable {
                worker->release_trace(cq_id, tid);
            });
    }
}

std::vector<ttnn::Tensor> get_device_tensors(const ttnn::Tensor& tensor) {
    if (std::holds_alternative<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage())) {
        std::vector<ttnn::Tensor> tensors;
        auto& host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage());
        for (int i = 0; i < host_storage.buffers.size(); ++i)
        {
            tensors.push_back(Tensor{OwnedStorage{host_storage.buffers[i]},  host_storage.shapes[i], tensor.get_dtype(), tensor.get_layout()});
        }
        return tensors;
    } else if (std::holds_alternative<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage())) {
        std::vector<ttnn::Tensor> tensors;
        auto& device_storage = std::get<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage());
        auto devices = get_devices(tensor);
        for (auto device : devices) {
            auto shard = get_shard_for_device(tensor, device);
            tensors.push_back(shard);
        }
        return tensors;
    }
    TT_THROW("Expected tensor to be on MultiDeviceHostStorage type!");
}

Tensor aggregate_as_tensor(std::vector<Tensor>& tensor_shards)
{
    TT_ASSERT(tensor_shards.size() > 0, "At least one tensor shard must be provided");
    for (const auto &shard : tensor_shards) {
        if (shard.storage_type() != tensor_shards.at(0).storage_type()) {
            TT_THROW("All tensor shards must have the same storage type");
        }
    }

    // Based whether the first tensor shard has OwnedBuffer or Device buffer,
    // we want to use MultiDeviceHostStorage or MultiDeviceStorage
    StorageType storage_type = tensor_shards.at(0).storage_type();
    if (storage_type == StorageType::OWNED) {
        std::vector<tt::tt_metal::Shape> shapes;
        std::vector<OwnedBuffer> host_owned_buffers;
        for (const auto &shard : tensor_shards) {
            host_owned_buffers.push_back(std::get<OwnedStorage>(shard.get_storage()).buffer);
            shapes.push_back(shard.get_legacy_shape());
        }
        auto storage = MultiDeviceHostStorage{AllGatherTensor(), std::move(host_owned_buffers), shapes};
        return Tensor(std::move(storage), tensor_shards.at(0).get_legacy_shape(), tensor_shards.at(0).get_dtype(),  tensor_shards.at(0).get_layout());
    } else {
        std::vector<int> ordered_device_ids;
        std::unordered_map<int, tt::tt_metal::Shape> shapes;
        std::unordered_map<int, DeviceBuffer> device_buffers;
        for (const auto &shard : tensor_shards) {
            Device* device = std::get<DeviceStorage>(shard.get_storage()).buffer->device();
            auto device_id = device->id();
            ordered_device_ids.push_back(device_id);
            device_buffers.insert({device->id(), std::get<DeviceStorage>(shard.get_storage()).buffer});
            shapes.insert({device->id(), shard.get_legacy_shape()});
        }
        auto storage = MultiDeviceStorage{AllGatherTensor(), ordered_device_ids, std::move(device_buffers), shapes};
        return Tensor(std::move(storage), tensor_shards.at(0).get_legacy_shape(), tensor_shards.at(0).get_dtype(),  tensor_shards.at(0).get_layout());
    }
}

}  // namespace multi_device

}  // namespace ttnn
