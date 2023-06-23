#include "debug_tools.hpp"

namespace internal {

void match_device_program_data_with_host_program_data(const char* host_file, const char* device_file) {
    std::ifstream host_dispatch_dump_file;
    std::ifstream device_dispatch_dump_file;

    host_dispatch_dump_file.open(host_file);
    device_dispatch_dump_file.open(device_file);

    string line;

    vector<pair<string, vector<i32>>> host_map;

    string type;
    while (not host_dispatch_dump_file.eof()) {
        std::getline(host_dispatch_dump_file, line);
        if (line.find("*")) continue;
        if (line.find("BINARY SPAN") or line.find("SEM") or line.find("CB")) {
            type = line;
        } else {
            vector<i32> host_data = {std::stoi(line)};
            while (not host_dispatch_dump_file.eof() and not line.find("*")) {
                std::getline(host_dispatch_dump_file, line);
                host_data.push_back(std::stoi(line));
            }
            host_map.push_back(make_pair(type, std::move(host_data)));
        }
    }

    vector<vector<i32>> device_map;
    while (not device_dispatch_dump_file.eof()) {
        std::getline(device_dispatch_dump_file, line);
        if (line == "CHUNK") {
            vector<i32> device_data;
            do {
                std::getline(device_dispatch_dump_file, line);
                device_data.push_back(std::stoi(line));
            } while (not device_dispatch_dump_file.eof() and line != "CHUNK");
            device_map.push_back(std::move(device_data));
        }
    }

    bool all_match = true;
    for (const auto&[type, host_data]: host_map) {
        bool match = false;
        for (const vector<i32>& device_data: device_map) {
            if (host_data == device_data) {
                match = true;
                break;
            }
        }

        if (not match) {
            tt::log_info("Mismatch between host and device program data on {}", type);
        }
        all_match &= match;
    }

    host_dispatch_dump_file.close();
    device_dispatch_dump_file.close();

    if (all_match) {
        tt::log_info("Full match between host and device program data");
    }

    exit(0);
}

void wait_for_program_vector_to_arrive_and_compare_to_host_program_vector(const char *DISPATCH_MAP_DUMP, Device* device) {
    std::string device_dispatch_dump_file_name = "device_" + std::string(DISPATCH_MAP_DUMP);
    while (true) {
        std::ifstream device_dispatch_dump_file;
        device_dispatch_dump_file.open(device_dispatch_dump_file_name);
        std::string line;
        while (!device_dispatch_dump_file.eof()) {

            std::getline(device_dispatch_dump_file, line);
            if (line.find("EXIT_CONDITION")) {
                device_dispatch_dump_file.close();

                CloseDevice(device);
                match_device_program_data_with_host_program_data(DISPATCH_MAP_DUMP, device_dispatch_dump_file_name.c_str());
            }
        }
    }
}

} // end namespace internal
