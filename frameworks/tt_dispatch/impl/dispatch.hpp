#include <queue>
#include <condition_variable>

#include "tt_metal/host_api.hpp"
#include "frameworks/tt_dispatch/impl/copy_descriptor.hpp"


enum class EventType: uint8_t {
    ADD_READ = 0,
    ADD_WRITE = 1,
    CLEAR = 2,
    WRITE_TO_SYSTEM_MEM = 3,
    ENQUEUE = 4
};

class Event {
    public:
        Event(EventType etype);
        void handle();

    private:
        EventType etype;
};

class DispatchManager;

class EventQueue {
    public:
        EventQueue(uint32_t capacity);

        void push(Event e);

        Event pop();

        size_t size();

    private:
        friend class DispatchManager;
        std::queue<Event> q;

        std::mutex m;
        std::condition_variable empty_condition;
        std::condition_variable full_condition;

        uint32_t capacity;
};

void run_worker(EventQueue &q);

class DispatchManager {
    public:
        DispatchManager(Device* device, uint32_t num_tables, uint32_t table_size_in_bytes);

        void push(Event &e);

        ~DispatchManager();

    private:
        // CopyDescriptor cd;
        EventQueue q;
        std::thread worker;
        std::condition_variable queue_flushed_condition;

};
