#include <iostream>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <vector>
#include <limits.h>
#include <unistd.h>
#include <sys/resource.h>
#include <cstdlib>
#include <time.h>

// compile with g++ -o use_cores use_cores.cpp -lpthread -O0

int get_random_priority(bool pos) {
    int p = rand() % 20;
    if (pos) {
        return p;
    }
    else {
        return p - 19;
    }
}

void heavy_work(int thread_id) {
    uint64_t num = 0;
    uint64_t upper_bound = 1e9;
    for (uint64_t i = 0; i < upper_bound; i++) {
        num += i;
    }
    std::cout << "Thread " << thread_id << " finished heavy work" << std::endl;
}

void set_cpu_affinity(std::thread& t, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc) {
        std::cout << "Unable to bind worker thread to CPU Core with return code " << rc << std::endl;
        exit(1);
    }
}

// void set_thread_priority(std::thread& t) {
//     int policy = -1;
//     struct sched_param param;
//     pthread_getschedparam(t.native_handle(), &policy, &param);
//     param.sched_priority = 10;
//     int rc = pthread_setschedparam(t.native_handle(), SCHED_RR, &param);
//     if (rc != 0) {
//         std::cout << "Can't set priority, return code: " << rc << std::endl;
//         exit(1);
//     }
//     struct sched_param param_new;
//     pthread_getschedparam(t.native_handle(), &policy, &param_new);
//     std::cout << "Policy: " << policy << std::endl;
//     std::cout << "Updated thread priority to: " << param.sched_priority << std::endl;
//     // if (rc != 0) {
//     //     std::cerr << "Error setting thread priority with return code " << rc << std::endl;
//     //     exit(1);
//     // }
// }

int main() {
    srand(time(NULL));
    static int num_online_cores = sysconf(_SC_NPROCESSORS_ONLN);
    bool positive = true;
    while (true) {
        int my_priority = getpriority(PRIO_PROCESS, 0);
        std::cout << "My priority intially: " << my_priority << std::endl;
        int rc = setpriority(PRIO_PROCESS, 0, get_random_priority(positive));
        positive = !positive;
        if (rc) {
            std::cout << "Could not set priority with return code: " << rc << std::endl;
        }
        else {
            std::cout << "Successfully set priority to: " << getpriority(PRIO_PROCESS, 0) << std::endl;
        }

        std::vector<std::thread> threads;
        for (unsigned int i = 0; i < 6; i++) {
            threads.push_back(std::thread(heavy_work, i));
            set_cpu_affinity(threads.back(), i);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        std::cout << "All threads finished heavy work" << std::endl;
    }

    return 0;
}
