// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <valarray>
#include "tt_metal/common/assert.hpp"
#include <iostream>

namespace tt {

namespace tt_metal {
class FunctionalLockFreeQueue {
    private:
        struct Node {
            std::function<void()> executor;
            Node* next = nullptr;
        };

        std::atomic<Node*> head;
        std::atomic<Node*> tail;
        Node* pop_head() {
            Node* oldHead = head.load();
            if (oldHead == tail.load()) {
                return nullptr; // Queue is empty
            }
            head.store(oldHead->next);
            return oldHead;
        }

    public:
        std::atomic<uint64_t> worker_thread_id = 0;
        std::atomic<uint64_t> parent_thread_id = 0;
        FunctionalLockFreeQueue() : head(new Node), tail(head.load()) {}
        FunctionalLockFreeQueue(FunctionalLockFreeQueue&& other) {
            std::cout << "calling move constructor on lock free queue" << std::endl;
            head.store(other.head.load());
            tail.store(other.tail.load());
            worker_thread_id.store(other.worker_thread_id.load());
            parent_thread_id.store(other.parent_thread_id.load());
        }

        void push(std::function<void()> work) {
            Node* newNode = new Node;
            tail.load()->executor = work;
            tail.load()->next = newNode;
            tail.store(newNode);
        }

        std::function<void()> pop() {
            Node* oldHead = pop_head();
            if (!oldHead) {
                TT_THROW("Queue is empty");
            }
            std::function<void()> result(oldHead->executor);
            delete oldHead;
            return result;
        }

        bool empty() const {
            return head.load() == tail.load();
        }
        class Iterator : public std::iterator<std::forward_iterator_tag, std::function<void()>> {
           private:
            Node* current;

           public:
            Iterator(Node* start) : current(start) {}

            Iterator& operator++() {
                if (current != nullptr) {
                    current = current->next;
                }
                return *this;
            }

            bool operator!=(const Iterator& other) const { return current != other.current; }

            const std::function<void()>& operator*() const { return current->executor; }
        };

        Iterator begin() { return Iterator(head.load()); }
        Iterator end() { return Iterator(tail.load()); }
};

}
}
