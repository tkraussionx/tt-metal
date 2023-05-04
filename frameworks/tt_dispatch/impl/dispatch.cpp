// #include "frameworks/tt_dispatch/impl/dispatch.hpp"
// #include <mutex>
// #include <thread>

// template <size_t T>
// Event<T>::Event(EventType etype): etype(etype) {}

// template <size_t T>
// void Event<T>::handle(EventArgsQueues<T> &event_args) {
//     switch (this->etype) {
//         case EventType::ADD_READ:
//             std::get<uint8_t(EventType::ADD_READ)>(event_args).pop();
//             break;
//         case EventType::ADD_WRITE:
//             std::get<uint8_t(EventType::ADD_WRITE)>(event_args).pop();
//             break;
//         case EventType::CLEAR:
//             std::get<uint8_t(EventType::CLEAR)>(event_args).pop();
//             break;
//         case EventType::WRITE_TO_SYSTEM_MEM:
//             std::get<uint8_t(EventType::WRITE_TO_SYSTEM_MEM)>(event_args).pop();
//             break;
//         case EventType::ENQUEUE:
//             std::get<uint8_t(EventType::ENQUEUE)>(event_args).pop();
//             break;
//         default:
//             TT_THROW("Invalid event type");
//     }
// }

// template <class T>
// TSQueue<T>::TSQueue(uint32_t capacity) {
//     this->q = std::queue<T>();
//     this->capacity = capacity;
// }

// template <class T>
// TSQueue<T>::TSQueue() {
//     this->q = std::queue<T>();
//     this->capacity = 100;
// }

// template <class T>
// void TSQueue<T>::push(T t) {

//     std::unique_lock<std::mutex> lock(this->m);

//     this->full_condition.wait(lock, [this]() { return this->q.size() < this->capacity; });

//     this->q.push(t);

//     this->empty_condition.notify_one();
// }

// template <class T>
// T TSQueue<T>::pop() {
//     std::unique_lock<std::mutex> lock(this->m);

//     this->empty_condition.wait(lock, [this]() { return !this->q.empty(); });

//     T t = this->q.front();
//     this->q.pop();

//     this->full_condition.notify_one();
//     return t;
// }

// template <class T>
// size_t TSQueue<T>::size() { return this->q.size(); }

// template <size_t T>
// void run_worker(EventQueue<T> &q, EventArgsQueues<T> &event_args) {
//     while (true) {
//         Event e = q.pop();
//         e.handle(event_args);
//     }
// }

// template <size_t T>
// DispatchManager<T>::DispatchManager(tt::tt_metal::Device* device, uint32_t num_tables, uint32_t table_size_in_bytes) {
//     this->worker = std::thread(run_worker, this->eventq, this->event_argsq);
//     this->worker.detach(); // Once main thread completes, auto-terminates this thread
// }

// template <size_t T>
// void DispatchManager<T>::push(Event<T> &e) {
//     this->eventq.push(e);
// }

// template <size_t T>
// void DispatchManager<T>::push_add_read_event_args(Event<T> &e, uint64_t src, uint32_t dst, uint32_t size_in_bytes) {
//     std::get<uint8_t(EventType::ADD_READ)>(this->event_argsq).push(tuple(this->cd, src, dst, size_in_bytes));
// }

// template <size_t T>
// void DispatchManager<T>::push_add_write_event_args(Event<T> &e, uint32_t src, uint64_t dst, uint32_t size_in_bytes) {
//     std::get<uint8_t(EventType::ADD_WRITE)>(this->event_argsq).push(tuple(this->cd, src, dst, size_in_bytes));
// }

// template <size_t T>
// void DispatchManager<T>::push_clear_event_args(Event<T> &e) {
//     std::get<uint8_t(EventType::CLEAR)>(this->event_argsq).push(this->cd);
// }

// template <size_t T>
// void DispatchManager<T>::push_write_to_system_mem_event_args(Event<T> &e, vector<uint32_t> vec) {
//     std::get<uint8_t(EventType::WRITE_TO_SYSTEM_MEM)>(this->event_argsq).push(vec);
// }

// template <size_t T>
// void DispatchManager<T>::push_enqueue_event_args(Event<T> &e) {
//     std::get<uint8_t(EventType::ENQUEUE)>(this->event_argsq).push(cd);
// }


// template <size_t T>
// DispatchManager<T>::~DispatchManager() {
//     std::mutex m;
//     std::unique_lock<std::mutex> lock(m);
//     this->eventq.full_condition.wait(lock, [this]() { return this->q.size() == 0; });
// }
