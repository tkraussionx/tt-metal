// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <unordered_set>
#include <queue>
#include <functional>
#include <chrono>
#include "assert.hpp"

struct ProgramThreadPool {
    public:

    ProgramThreadPool() {
         this->total_num_workers_ = std::thread::hardware_concurrency();
        this->toggle_enable_cpu_affinity(true);
        this->start();
    }

    ProgramThreadPool(uint32_t num_threads) {
        TT_ASSERT(num_threads >= 1, "Number of threads must be greater than or equal to 1");
        this->toggle_enable_cpu_affinity(true);
        this->total_num_workers_ = num_threads;
        this->start();
    }

    void toggle_enable_cpu_affinity(bool enable) {
        TT_ASSERT(this->is_stopped(), "Cannot toggle thread affinity after starting the thread pool");
        this->cpu_affinity_ = enable;
    }

    void start() {
        TT_ASSERT(this->is_stopped(), "Please stop the thread pool before starting a new one");
        this->num_free_workers_ = this->total_num_workers_;
        this->stop_flag_ = false;
        for (uint32_t i = 0; i < this->total_num_workers_; i++) {
            this->threads_.push_back(std::thread(&ProgramThreadPool::loop, this));
            this->thread_ids_.insert(this->threads_.back().get_id());

            if (this->cpu_affinity_) {
                // Bind a worker tied to a device to a specific CPU core in round robin fashion. Thread affinity == Better Perf.
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(i % std::thread::hardware_concurrency(), &cpuset);
                int rc = pthread_setaffinity_np(this->threads_.back().native_handle(), sizeof(cpu_set_t), &cpuset);
                TT_ASSERT(rc == 0, "Unable to bind worker thread to CPU Core");
            }
        }
        TT_ASSERT(this->threads_.size() == this->thread_ids_.size(), "Unexpected number of threads created does not match number of thread ids");
    }

    void queue_job(const std::function<void()>& push_func, bool force_fork_thread = false) {
        std::thread::id thread_id = std::this_thread::get_id();
        TT_ASSERT(!this->is_stopped(), "Cannot queue job after stopping the thread pool");
        // If the thread is not in the thread pool, then fork a new thread
        // If the thread is in the thread pool and we don't have any free workers, then execute the job directly
        if (this->num_free_workers_.load() == 0 and this->thread_ids_.find(thread_id) != this->thread_ids_.end() and !force_fork_thread) {
            push_func();
            return;
        }
        this->jobs_q_.push(push_func);
        this->job_cv_.notify_one();
    }

    void stop() {
        TT_ASSERT(!this->is_stopped(), "Cannot stop a already-stopped thread pool");
        this->stop_flag_ = true;
        this->job_cv_.notify_all();
        for (std::thread& active_thread : this->threads_) {
            active_thread.join();
        }
        this->threads_.clear();
        this->thread_ids_.clear();
    }

    bool is_stopped() const {
        return this->stop_flag_.load();
    }

    bool has_free_workers() const {
        TT_ASSERT(!this->is_stopped(), "Cannot get free workers after stopping the thread pool");
        return this->num_free_workers_.load() > 0;
    }

    bool all_done() const {
        TT_ASSERT(!this->is_stopped(), "Cannot check if all jobs are done after stopping the thread pool");
        return (this->num_free_workers_.load() == this->threads_.size()) and (this->jobs_q_.empty());
    }

    void sync() const {
        TT_ASSERT(!this->is_stopped(), "Cannot sync after stopping the thread pool");
        while (!this->all_done()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    private:
    bool cpu_affinity_ = false;
    uint32_t total_num_workers_;
    std::atomic<bool> stop_flag_ = true;
    std::atomic<uint32_t> num_free_workers_ = 0;
    std::mutex core_mutex;
    std::mutex job_mutex_;
    std::condition_variable job_cv_;
    std::unordered_set<std::thread::id> thread_ids_;
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> jobs_q_;
    void loop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(job_mutex_);
                this->job_cv_.wait(lock, [this] { return this->is_stopped() or !(this->jobs_q_.empty());});
                if (this->is_stopped()) return;
                this->num_free_workers_--;
                job = this->jobs_q_.front();
                this->jobs_q_.pop();
            }
            job();
            this->num_free_workers_++;
        }
    }
        
};