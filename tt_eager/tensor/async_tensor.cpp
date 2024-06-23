// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/async_tensor.hpp"

#include <cstdint>
#include <memory>

#include "common/bfloat16.hpp"
#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_impl_wrapper.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/types.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"


namespace tt {

namespace tt_metal {

/* ============================================= TensorShard Implementation  ============================================= */
TensorShard::TensorShard(const Storage& storage, const ttnn::Shape& shape, DataType dtype, Layout layout) : storage(storage), shape(shape), dtype(dtype), layout(layout) {
    // Tensor is populated.
    this->populated = true;
}

TensorShard::TensorShard(const Storage storage, const Shape shape, DataType dtype, Layout layout) : TensorShard(storage, ttnn::Shape(shape), dtype, layout) {};

TensorShard::TensorShard(const TensorShard &other) : storage(other.storage), shape(other.shape), dtype(other.dtype), layout(other.layout) {
    // Copied tensor is populated if reference tensor is populated
    this->populated.store(other.populated.load());
}

TensorShard::~TensorShard() {
    this->deallocate();
}

void TensorShard::deallocate(bool force) {
    if ((not this->deallocated) and this->populated) {
        // Run deallocate function if tensor was not previously deallocated or if tensor
        // is populated.
        std::visit([force] (auto&& storage) {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                std::visit([](auto && buffer) { buffer.reset(); }, storage.buffer);
            } else if constexpr (std::is_same_v<T, DeviceStorage>) {
                if (force or storage.buffer.use_count() == 1) {
                    DeallocateBuffer(*(storage.buffer));
                }
                storage.buffer.reset();
            } else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                TT_FATAL(!force, "Cannot deallocate tensor with borrowed storage!");
            } else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                return;
            } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                return;
            } else {
                raise_unsupported_storage<T>();
            }
        }, this->storage);
        this->deallocated = true;
    }
}

TensorShard TensorShard::to(Device* target_device, const MemoryConfig& mem_cfg) const {
    ZoneScoped;
    if (this->storage_type() == StorageType::DEVICE) {
        TT_ASSERT(this->device() == target_device && "Currently do not support moving between devices");
        return *this;
    }
    tensor_impl::validate_on_device_dtype_and_layout(
        target_device,
        this->get_legacy_shape(),
        this->get_dtype(),
        this->get_layout());
    return tensor_impl::to_device_wrapper(*this, target_device, mem_cfg, std::nullopt);
}

TensorShard TensorShard::cpu(bool blocking) const {
    ZoneScoped;
    if (this->storage_type() == StorageType::DEVICE) {
        return tensor_impl::to_host_wrapper(*this, blocking);
    }
    return *this;
}

/* ============================================= AsyncTensor Implementation  ============================================= */
AsyncTensor::AsyncTensor(const std::vector<Device*>& workers, uint32_t num_shards) {
    if (workers.size()) {
        this->workers = workers;
        this->num_shards_to_be_populated = this->workers.size();
        for (const auto& worker : this->workers) {
            // Initialize shared_ptrs to empty shards
            shards.insert({worker->id(), std::make_shared<TensorShard>()});
        }
    } else {
        this->num_shards_to_be_populated = num_shards;
        for (std::size_t shard_idx = 0; shard_idx < this->num_shards_to_be_populated; shard_idx++) {
            // Initialize shared_ptrs to empty shards
            shards.insert({shard_idx, std::make_shared<TensorShard>()});
        }
    }
    // A single tensor in the main thread exists with these attributes.
    (*(this->main_thread_ref_count))++;
}

AsyncTensor::AsyncTensor(const Storage& storage, const ttnn::Shape& shape, DataType dtype, Layout layout) {
    auto shard = std::make_shared<TensorShard>(storage, shape, dtype, layout);
    this->shards.insert({0, shard});
    this->num_workers_completed = 1;
    this->num_shards_to_be_populated = 1;
    this->metadata_idx = 0;
    if (std::holds_alternative<DeviceStorage>(storage)) {
        this->workers = {std::get<DeviceStorage>(storage).get_buffer()->device()};
    }
    // A single tensor in the main thread exists with these attributes.
    (*(this->main_thread_ref_count))++;
}

AsyncTensor::AsyncTensor(const std::unordered_map<uint32_t, Storage>& storages, const std::unordered_map<uint32_t, ttnn::Shape> shapes, DataType dtype, Layout layout) {
    for (const auto& shard_storage : storages) {
        auto shard = std::make_shared<TensorShard>(shard_storage.second, shapes.at(shard_storage.first), dtype, layout);
        this->shards.insert({shard_storage.first, shard});
        if (std::holds_alternative<DeviceStorage>(shard_storage.second)) {
            if (not this->workers.size()) {
                this->workers = std::vector<Device*>(storages.size());
            }
            this->workers.at(shard_storage.first) = std::get<DeviceStorage>(shard_storage.second).get_buffer()->device();
        }
    }
    this->num_shards_to_be_populated = storages.size();
    this->num_workers_completed = storages.size();
    this->metadata_idx = 0;
    // A single tensor in the main thread exists with these attributes.
    (*(this->main_thread_ref_count))++;
}

AsyncTensor::~AsyncTensor() {
    this->deallocate();
    if (this->main_thread_ref_count) {
        // Decrease the number of tensors that exist with these attributes.
        (*(this->main_thread_ref_count))--;
        this->main_thread_ref_count.reset();
    }
}

void AsyncTensor::deallocate(bool force) {
    if (this->main_thread_ref_count and *(this->main_thread_ref_count)) {
        // Tensor has an associated ref count (was not previously deallocated by copy or move).
        if (this->workers.size()) {
            // Tensor is on device.
            if (*(this->main_thread_ref_count) == 1) {
                // Last tensor that holds these shards, deallocate them through worker.
                for (auto worker : this->workers) {
                    worker->push_work([shard_to_deallocate = this->get_shard_at_idx(worker->id())] () {
                        shard_to_deallocate->deallocate();
                    });
                }
            }
        } else {
            // Host buffer deallocation Logic - Give up shared ptr to host storage.
            for (auto &shard : shards) {
                shard.second.reset();
            }
        }
    }
}

AsyncTensor AsyncTensor::to(Device* device, const MemoryConfig& mem_cfg) const {
    AsyncTensor device_tensor({device});
    device->push_work([device, host_shard = this->get_shard_at_idx(0), device_shard = device_tensor.get_shard_at_idx(device->id())] () mutable {
        *device_shard = host_shard->to(device);
    });
    return device_tensor;
}

AsyncTensor AsyncTensor::cpu(bool blocking) const {
    AsyncTensor host_tensor({}, this->workers.size());
    this->workers.at(0)->push_work([device_shard = this->get_shard_at_idx(0), host_shard = host_tensor.get_shard_at_idx(0)] () mutable {
        auto local_tensor = device_shard->cpu();
        *host_shard = device_shard->cpu();
    });
    return host_tensor;
}

const Storage& AsyncTensor::get_storage(int storage_idx) const {
    // Get storage at specific idx.
    wait_for_storage_attributes_populated();
    return this->shards.at(storage_idx)->get_storage();
}

std::unordered_map<int, Storage> AsyncTensor::get_storage() const {
    // Aggregate storage across all shards -> represents multi_device storage.
    wait_for_storage_attributes_populated();
    std::unordered_map<int, Storage> storage_per_shard = {};
    for (const auto& shard : this->shards) {
        storage_per_shard.insert({shard.first, shard.second->get_storage()});
    }
    return storage_per_shard;
}

const ttnn::Shape& AsyncTensor::get_shape() const {
    wait_for_metadata_populated();
    // Can aggregate the shapes here if required
    return this->shards.at(this->metadata_idx)->get_shape();
}

const DataType& AsyncTensor::get_dtype() const {
    wait_for_metadata_populated();
    return this->shards.at(this->metadata_idx)->get_dtype();
}

const Layout& AsyncTensor::get_layout() const {
    wait_for_metadata_populated();
    return this->shards.at(this->metadata_idx)->get_layout();
}

std::shared_ptr<TensorShard> AsyncTensor::get_shard_at_idx(int idx) const {
    return shards.at(idx);
}

DeviceBuffer AsyncTensor::device_buffer() const {
    wait_for_storage_attributes_populated();
    return this->shards.at(0)->device_buffer();
}

uint32_t AsyncTensor::get_ref_count() const {
    return *(this->main_thread_ref_count);
}

} // namespace tt_metal

} // namespace tt
