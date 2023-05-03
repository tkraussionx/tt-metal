#include <queue>
#include <condition_variable>
#include <thread>

#include "tt_metal/host_api.hpp"
#include "frameworks/tt_dispatch/impl/copy_descriptor.hpp"

using std::tuple;

enum class EventType: uint8_t {
    ADD_READ,
    ADD_WRITE,
    CLEAR,
    WRITE_TO_SYSTEM_MEM,
    ENQUEUE,

    // Only used to get the number of elements in the enum, not
    // an actual event
    COUNT
};

class Event {
    public:
        Event(EventType etype);
        void handle();

    private:
        EventType etype;
};

template <size_t T>
class DispatchManager;

template <class T>
class TSQueue {
    public:
        TSQueue();
        TSQueue(uint32_t capacity);

        void push(T e);

        T pop();

        size_t size();
        std::queue<T> q;
        std::condition_variable empty_condition;
        std::condition_variable full_condition;
        uint32_t capacity;

    private:
        std::mutex m;
};

typedef TSQueue<Event> EventQueue;

// Make a queue containing arguments for each of the possible events
template <size_t T>
using AddReadArgQueue = TSQueue<tuple<
    CopyDescriptor<T>&,
    uint64_t, // src
    uint32_t, // dst
    uint32_t // size in bytes
>>;

template <size_t T>
using AddWriteArgQueue = TSQueue<tuple<
    CopyDescriptor<T>&,
    uint32_t,
    uint64_t,
    uint32_t
>>;

using WriteToSystemMemArgQueue = TSQueue<std::vector<uint32_t>>;

// // void run_worker(EventQueue &q);

template <size_t T>
using EventArgsQueues = std::tuple<
    AddReadArgQueue<T>, AddWriteArgQueue<T>, WriteToSystemMemArgQueue
>;


template <size_t T>
class DispatchManager {
    public:
        DispatchManager(tt::tt_metal::Device *device, uint32_t num_tables, uint32_t table_size_in_bytes);

        void push(Event &e);

        ~DispatchManager();

    private:
        CopyDescriptor<T> cd;
        EventQueue eventq;

        EventArgsQueues<T> event_argsq;

        std::thread worker;
};
