#include "frameworks/tt_dispatch/impl/dispatch.hpp"
#include <mutex>
#include <thread>

Event::Event(EventType etype): etype(etype) {}

void Event::handle() {
    switch (this->etype) {
        case EventType::ADD_READ:

            break;
        case EventType::ADD_WRITE:
            break;
        case EventType::CLEAR:
            break;
        case EventType::WRITE_TO_SYSTEM_MEM:
            break;
        case EventType::ENQUEUE:
            break;
        default:
            TT_THROW("Invalid event type");
    }
}

EventQueue::EventQueue(uint32_t capacity) {
    this->q = std::queue<Event>();
    this->capacity = capacity;
}

void EventQueue::push(Event e) {

    std::unique_lock<std::mutex> lock(this->m);

    this->full_condition.wait(lock, [this]() { return this->q.size() < this->capacity; });

    this->q.push(e);

    this->empty_condition.notify_one();
}

Event EventQueue::pop() {
    std::unique_lock<std::mutex> lock(this->m);

    this->empty_condition.wait(lock, [this]() { return !this->q.empty(); });

    Event e = this->q.front();
    this->q.pop();

    this->full_condition.notify_one();
    return e;
}

size_t EventQueue::size() { return this->q.size(); }

void run_worker(EventQueue &q) {
    while (true) {
        Event e = q.pop();
        e.handle();
    }
}

void DispatchManager::push(Event &e) { this->q.push(e); }

DispatchManager::DispatchManager(Device* device, uint32_t num_tables, uint32_t table_size_in_bytes) {
    this->worker = std::thread(run_worker, this->q);
    this->worker.detach(); // Once main thread completes, auto-terminates this thread
}

DispatchManager::~DispatchManager() {
    std::mutex m;
    std::unique_lock<std::mutex> lock(m);
    this->queue_flushed_condition.wait(lock, [this]() { return this->q.size() == 0; });
}
