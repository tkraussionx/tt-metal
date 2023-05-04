#include <queue>
#include <condition_variable>
#include <thread>

#include "tt_metal/host_api.hpp"
#include "frameworks/tt_dispatch/impl/copy_descriptor.hpp"

using std::tuple;

using namespace tt::tt_metal;

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

// template <size_t T>
// using EventArgsQueues = std::tuple<
//     AddReadArgQueue<T>, AddWriteArgQueue<T>, WriteToSystemMemArgQueue
// >;

template <size_t T>
class DispatchManager;

#include <mutex>
#include <thread>

template <class T>
TSQueue<T>::TSQueue(uint32_t capacity) {
    this->q = std::queue<T>();
    this->capacity = capacity;
}

template <class T>
TSQueue<T>::TSQueue() {
    this->q = std::queue<T>();
    this->capacity = 100;
}

template <class T>
void TSQueue<T>::push(T t) {

    std::unique_lock<std::mutex> lock(this->m);

    this->full_condition.wait(lock, [this]() { return this->q.size() < this->capacity; });

    this->q.push(t);

    this->empty_condition.notify_one();
}

template <class T>
T TSQueue<T>::pop() {
    std::unique_lock<std::mutex> lock(this->m);

    this->empty_condition.wait(lock, [this]() { return !this->q.empty(); });

    T t = this->q.front();
    this->q.pop();

    this->full_condition.notify_one();
    return t;
}

template <class T>
size_t TSQueue<T>::size() { return this->q.size(); }

// template <size_t T>
// void run_worker(EventQueue<T> &q, EventArgsQueues<T> &event_args) {
//     while (true) {
//         Event e = q.pop();
//         e.handle(event_args);
//     }
// }

enum class CommandType: uint {
    ENQUEUE_PROGRAM,
    ENQUEUE_WRITE_DRAM_BUFFER, // For now, but eventually would want to make this generic (write buffer, rather than DRAM buffer)
    ENQUEUE_READ_DRAM_BUFFER,
};

class Command {
    public:
        Command() {}
        virtual uint handle() = 0;
};

class EnqueueProgramCommand: public Command {
    public:
        EnqueueProgramCommand();
        uint handle();
};

uint EnqueueProgramCommand::handle() {
    return uint(CommandType::ENQUEUE_PROGRAM);
}

class EnqueueWriteBufferCommand: public Command {
    public:
        EnqueueWriteBufferCommand();
        uint handle();
};

uint EnqueueWriteBufferCommand::handle() {
    return uint(CommandType::ENQUEUE_WRITE_DRAM_BUFFER);
}

class EnqueueReadBufferCommand: public Command {
    public:
        EnqueueReadBufferCommand();
        uint handle();
};

uint EnqueueReadBufferCommand::handle() {
    return uint(CommandType::ENQUEUE_READ_DRAM_BUFFER);
}

using CommandQueue = TSQueue<Command*>;


void EnqueueProgram(
    tt::tt_metal::Device* device,
    CommandQueue &command_queue,
    Program* program,
    bool blocking
) {
    if (blocking) {
        // command_queue.push(&command);
    } else {
        TT_THROW("Non-blocking EnqueueProgram not yet supported");
    }
}

void EnqueueWriteToDeviceDRAM(
    tt::tt_metal::Device* device,
    CommandQueue &command_queue,
    DramBuffer* buffer,
    void* src,
    bool blocking
) {
    EnqueueWriteBufferCommand command;
    if (blocking) {
        command_queue.push(&command);

    } else {
        TT_THROW("Non-blocking EnqueueWriteBuffer not yet supported");
    }
}

void EnqueueReadFromDeviceDRAM(
    tt::tt_metal::Device* device,
    CommandQueue &command_queue,
    DramBuffer* buffer,
    void* dst,
    bool blocking
) {
    if (blocking) {
        // command_queue.push(&command);
    } else {
        TT_THROW("Non-blocking EnqueueWriteBuffer not yet supported");
    }
}
