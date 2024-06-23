// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor.hpp"

namespace tt {

namespace tt_metal {

struct TensorShard {
    Storage storage;
    ttnn::Shape shape;
    DataType dtype;
    Layout layout;
    std::atomic<bool> populated = false;
    bool deallocated = false;
    TensorShard(const Storage& storage, const ttnn::Shape& shape, DataType dtype, Layout layout);

    TensorShard(const Storage storage, const Shape shape, DataType dtype, Layout layout);

    TensorShard(const TensorShard &other);

    TensorShard() : shape(std::array<uint32_t, 4>{0xff, 0xff, 0xff, 0xff}), dtype(DataType::INVALID), layout(Layout::INVALID) {};

    ~TensorShard();

    TensorShard &operator=(const TensorShard& other) {
        this->storage = other.storage;
        this->shape = other.shape;
        this->dtype = other.dtype;
        this->layout = other.layout;
        this->populated.store(other.populated.load());
        return *this;
    }

    const ttnn::Shape& get_shape() const { return this->shape; }
    const Shape& get_legacy_shape() const { return this->shape.value(); }
    const Storage& get_storage() const { return this->storage; }
    const DataType& get_dtype() const { return this->dtype; }
    const Layout& get_layout() const { return this->layout; }

    StorageType storage_type() const {
        return std::visit(
            [](auto&& storage) -> StorageType {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, OwnedStorage>) {
                    return StorageType::OWNED;
                } else if constexpr (std::is_same_v<T, DeviceStorage>) {
                    return StorageType::DEVICE;
                } else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                    return StorageType::BORROWED;
                } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                    return StorageType::MULTI_DEVICE;
                } else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                    return StorageType::MULTI_DEVICE_HOST;
                } else {
                    raise_unsupported_storage<T>();
                }
            }, this->get_storage());
    }

    Buffer *buffer() const { return std::get<DeviceStorage>(this->get_storage()).get_buffer().get(); }
    DeviceBuffer device_buffer() const { return std::get<DeviceStorage>(this->get_storage()).get_buffer(); }
    Device *device() const {
        if (this->storage_type() == tt::tt_metal::StorageType::DEVICE) {
            auto buffer = this->buffer();
            if (buffer == nullptr)
                TT_THROW("Cannot get the device from a tensor without an allocated buffer");
            return buffer->device();
        } else {
            TT_THROW("Cannot get the device from a tensor with host storage");
        }
    }

    void deallocate(bool force = false);

    TensorShard to(Device* target_device, const MemoryConfig& mem_cfg = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) const;

    TensorShard cpu(bool blocking = true) const;
};

struct AsyncTensor {
    private:
    // Tensor Shards
    std::unordered_map<uint32_t, std::shared_ptr<TensorShard>> shards = {};
    // Async engine handles.
    std::vector<Device*> workers = {};
    // Async metadata.
    mutable int metadata_idx = -1;
    mutable uint32_t num_workers_completed = 0;
    uint32_t num_shards_to_be_populated = 0;
    // Keeps track of how many instances of this Tensor exist in the main thread (Tensor objects should only exist in the main thread).
    // A shared ptr, since we want all tensors tied to a set of shards to maintain a shared value for the ref count.
    std::shared_ptr<uint32_t> main_thread_ref_count = std::make_shared<uint32_t>(0);

    public:
    AsyncTensor(const std::vector<Device*>& workers = {}, uint32_t num_shards = 0);

    AsyncTensor(const Storage& storage, const ttnn::Shape& shape, DataType dtype, Layout layout);

    AsyncTensor(const std::unordered_map<uint32_t, Storage>& storages, const std::unordered_map<uint32_t, ttnn::Shape> shapes, DataType dtype, Layout layout);

    // Cast a Tensor to an AsyncTensor.
    AsyncTensor(const Tensor& tensor) : AsyncTensor(tensor.get_storage(), tensor.get_shape(), tensor.get_dtype(), tensor.get_layout()) {};

    // Copy/Move constructors and operators.
    AsyncTensor (AsyncTensor &&other) = default;

    AsyncTensor &operator=(AsyncTensor &&other)  {
        if (this->main_thread_ref_count != other.main_thread_ref_count) {
            // Don't self assign.
            this->workers = std::move(other.workers);
            // This tensor will now contain new storage/buffers.
            // Perform required cleanup.
            if (*(this->main_thread_ref_count) == 1) {
                this->deallocate();
            }
            // Decrement the current ref count.
            (*(this->main_thread_ref_count))--;
            // Complete the assignment.
            this->shards = std::move(other.shards);
            this->metadata_idx = std::move(other.metadata_idx);
            this->num_workers_completed = std::move(other.num_workers_completed);
            this->num_shards_to_be_populated = std::move(other.num_shards_to_be_populated);
            this->main_thread_ref_count = std::move(other.main_thread_ref_count);
        }
        return *this;
    }

    AsyncTensor (const AsyncTensor &other) {
        this->shards = other.shards;
        this->workers = other.workers;
        this->metadata_idx = other.metadata_idx;
        this->num_workers_completed = other.num_workers_completed;
        this->num_shards_to_be_populated = other.num_shards_to_be_populated;
        this->main_thread_ref_count = other.main_thread_ref_count;
        // Increment the ref count of the tensor since it was copied. This update will be visible to all tensors
        // that are copies of this or other.
        (*(this->main_thread_ref_count))++;
    }

    AsyncTensor &operator=(const AsyncTensor &other)  {
        if (this->main_thread_ref_count != other.main_thread_ref_count) {
            // Don't self assign.
            this->workers = other.workers;
            // This tensor will now contain new storage/buffers.
            // Perform required cleanup -> deallocate the previously
            // owned storage object, if no other tensors are sharing it.
            if (*(this->main_thread_ref_count) == 1) {
                this->deallocate();
            }
            // Decrement the current ref count.
            (*(this->main_thread_ref_count))--;
            // Complete the assignment.
            this->shards = other.shards;
            this->metadata_idx = other.metadata_idx;
            this->num_workers_completed = other.num_workers_completed;
            this->num_shards_to_be_populated = other.num_shards_to_be_populated;
            this->main_thread_ref_count = other.main_thread_ref_count;
            // Increment the new ref count.
            (*(this->main_thread_ref_count))++;
        }
        return *this;
    }

    // Destructor
    ~AsyncTensor();

    inline void wait_for_metadata_populated() const {
        while (this->metadata_idx == -1) {
            for (auto& shard : this->shards) {
                if (shard.second->populated) {
                    this->metadata_idx = shard.first;
                    break;
                }
            }
        }
    }

    inline void wait_for_storage_attributes_populated() const {
        std::unordered_set<uint32_t> tagged_shards = {};
        while (this->num_workers_completed < num_shards_to_be_populated) {
            for (auto& shard : this->shards) {
                if (shard.second->populated and tagged_shards.find(shard.first) == tagged_shards.end()) {
                    tagged_shards.insert(shard.first);
                    this->num_workers_completed++;
                }
            }
        }
    }

    const Storage& get_storage(int storage_idx) const;

    std::unordered_map<int, Storage> get_storage() const;

    const ttnn::Shape& get_shape() const;

    const DataType& get_dtype() const;

    const Layout& get_layout() const;

    std::shared_ptr<TensorShard> get_shard_at_idx(int idx) const;

    DeviceBuffer device_buffer() const;

    uint32_t get_ref_count() const;

    AsyncTensor to(Device* device, const MemoryConfig& mem_cfg = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) const;

    AsyncTensor cpu(bool blocking = true) const;

    void deallocate(bool force = false);
};

} // namespace tt_metal

} // namespace tt
