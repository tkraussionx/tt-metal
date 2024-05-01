// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <functional>
#include <memory>

#include "third_party/concurrentqueue/concurrentqueue.h"
#include "tt_metal/common/assert.hpp"

template <typename T>
class LockFreeQueue {
   private:
    moodycamel::ConcurrentQueue<std::shared_ptr<T>> queue;

   public:
    // Optional - Set these if the worker and parent thread state needs to be tracked
    std::atomic<uint64_t> worker_thread_id = 0;
    std::atomic<uint64_t> parent_thread_id = 0;

    LockFreeQueue() = default;
    LockFreeQueue(const LockFreeQueue&) = delete;  // Disable copying
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;

    LockFreeQueue(LockFreeQueue&& other) : queue(std::move(other.queue)) {
        worker_thread_id.store(other.worker_thread_id.load());
        parent_thread_id.store(other.parent_thread_id.load());
    }
    void push(const T& value) { queue.enqueue(std::make_shared<T>(value)); }

    std::shared_ptr<T> pop() {
        std::shared_ptr<T> result;
        if (!queue.try_dequeue(result)) {
            TT_THROW("Queue is empty");
        }
        return result;
    }

    bool empty() const { return queue.size_approx() == 0; }
    // class Iterator {
    //    public:
    //     using iterator_category = std::forward_iterator_tag;
    //     using value_type = T;
    //     using difference_type = std::ptrdiff_t;
    //     using pointer = T*;
    //     using reference = T&;

    //    private:
    //     Node* current;

    //    public:
    //     // Constructor initializes the iterator with a pointer to a Node
    //     Iterator(Node* start) : current(start) {}

    //     // Prefix increment operator overloading
    //     Iterator& operator++() {
    //         if (current != nullptr) {
    //             current = current->next;
    //         }
    //         return *this;
    //     }

    //     // Inequality operator overloading
    //     bool operator!=(const Iterator& other) const { return current != other.current; }

    //     // Dereference operator overloading
    //     const T& operator*() const { return *(current->data); }
    // };

    // Iterator begin() { return Iterator(head.load()); }
    // Iterator end() { return Iterator(tail.load()); }
};
