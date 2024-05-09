// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include "tt_metal/common/assert.hpp"

/*
    Supports single writer, single reader
*/
template<typename T>
class LockFreeQueue {
    private:
        struct Node {
            std::shared_ptr<T> data = nullptr;
            Node* next = nullptr;
        };

        std::atomic<Node*> head;
        std::atomic<Node*> tail;

        inline Node* pop_head() {
            Node* oldHead = head.load();
            if (oldHead == tail.load()) {
                return nullptr; // Queue is empty
            }
            head.store(oldHead->next);
            return oldHead;
        }
        Node nodes[50000];
    public:
        // Optional - Set these if the worker and parent thread state needs to be tracked
        std::atomic<uint64_t> worker_thread_id = 0;
        std::atomic<uint64_t> parent_thread_id = 0;
        LockFreeQueue()
        {
            for (int i = 0; i < 50000; i++) {
                if (i < 49999) nodes[i].next = &(nodes[i+1]);
                else nodes[i].next = &(nodes[0]);
            }
            this->head = nodes;
            this->tail = nodes;
        }

        LockFreeQueue(LockFreeQueue&& other) {
            Node nodes = other.nodes;
            head.store(other.head.load());
            tail.store(other.tail.load());
            worker_thread_id.store(other.worker_thread_id.load());
            parent_thread_id.store(other.parent_thread_id.load());
        }
        inline void push(const T& value) {
            while(tail.load()->data != nullptr);
            tail.load()->data = std::make_shared<T>(value);
            tail.store(tail.load()->next);
        }

        inline void push(std::shared_ptr<T> value) {
            while(tail.load()->data != nullptr);
            tail.load()->data = value;
            tail.store(tail.load()->next);
        }

        inline std::shared_ptr<T> pop() {
            Node* oldHead = pop_head();
            std::shared_ptr<T> result(oldHead->data);
            (oldHead->data).reset();
            return result;
        }

        void clear() {
            while (!empty()) {
                void(pop());
            }
        }

        bool empty() const {
            return head.load() == tail.load();
        }
        class Iterator {
           public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = T;
            using difference_type = std::ptrdiff_t;
            using pointer = T*;
            using reference = T&;

           private:
            Node* current;

           public:
            // Constructor initializes the iterator with a pointer to a Node
            Iterator(Node* start) : current(start) {}

            // Prefix increment operator overloading
            Iterator& operator++() {
                if (current != nullptr) {
                    current = current->next;
                }
                return *this;
            }

            // Inequality operator overloading
            bool operator!=(const Iterator& other) const { return current != other.current; }

            // Dereference operator overloading
            const T& operator*() const { return *(current->data); }
        };

        Iterator begin() { return Iterator(head.load()); }
        Iterator end() { return Iterator(tail.load()); }
};
