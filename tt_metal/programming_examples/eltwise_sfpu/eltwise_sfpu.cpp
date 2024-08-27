// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/third_party/taskflow/3rd-party/httplib/httplib.hpp"

using namespace tt;
using namespace tt::tt_metal;

/*
* 1. Host creates one vector of data.
* 2. Device eltwise performs a unary SFPU operation on the data.
* 3. Read result back and compare to golden.
* */
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <chrono> // For time measurement
#include <openssl/md5.h>


std::string base64_encode(const std::string& in) {
    static const std::string base64_chars =
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                 "abcdefghijklmnopqrstuvwxyz"
                 "0123456789+/";
    std::string out;
    int val = 0, valb = -6;
    for (unsigned char c : in) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) out.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}


std::vector<long long> durations;

std::string send_get_request(const std::string& hostname, const std::string& port, const std::string& request1, const std::string& request2, int num_tries) {
    int sockfd;
    struct addrinfo hints, *res;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(hostname.c_str(), port.c_str(), &hints, &res) != 0) {
        throw std::runtime_error("getaddrinfo failed");
    }

    sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sockfd == -1) {
        freeaddrinfo(res);
        throw std::runtime_error("socket creation failed");
    }

    if (connect(sockfd, res->ai_addr, res->ai_addrlen) == -1) {
        close(sockfd);
        freeaddrinfo(res);
        throw std::runtime_error("connection failed");
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    int num_calls = 0;
    std::vector<char> response_buffer;
    for (int j=0, k=0;j<10;j++)
    {
        for (int i=0;i<num_tries;i++,k++)
        {
            response_buffer.clear();
            // Sending the request directly
            num_calls++;
            if (num_calls % 100 == 0)
            {
                close(sockfd);
                freeaddrinfo(res);
                if (getaddrinfo(hostname.c_str(), port.c_str(), &hints, &res) != 0) {
                    throw std::runtime_error("getaddrinfo failed");
                }
                sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
                if (sockfd == -1) {
                    freeaddrinfo(res);
                    throw std::runtime_error("socket creation failed");
                }
                if (connect(sockfd, res->ai_addr, res->ai_addrlen) == -1) {
                    close(sockfd);
                    freeaddrinfo(res);
                    throw std::runtime_error("connection failed");
                }
            }
            if (i%2==0)
            {
                send(sockfd, request1.c_str(), request1.size(), 0);
            }
            else
            {
                send(sockfd, request2.c_str(), request2.size(), 0);
            }
            char buffer[4096];
            int bytes_received;

            while ((bytes_received = recv(sockfd, buffer, sizeof(buffer), 0)) > 0) {
                response_buffer.clear();
                response_buffer.insert(response_buffer.end(), buffer, buffer + bytes_received);
                if (response_buffer[response_buffer.size()-5] == '0') break;
            }
        }
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto response_time2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end1).count();
    std::cout << "times=" << num_tries << std::endl;
    std::cout << "overall_duration=" << response_time2/10 << std::endl;
    durations.push_back(response_time2/10);
    close(sockfd);
    freeaddrinfo(res);

    std::string response_string(response_buffer.begin(), response_buffer.end());
    return response_string;
}

std::string extract_status(const std::string& json_response) {
    // Simple manual JSON parsing for demonstration purposes
    size_t status_pos = json_response.find("\"status\":\"");
    if (status_pos == std::string::npos) return "";

    status_pos += 10;  // Move past the "status":" part
    size_t end_pos = json_response.find("\"", status_pos);
    if (end_pos == std::string::npos) return "";

    return json_response.substr(status_pos, end_pos - status_pos);
}

std::string to_hash(const std::string& input) {
    // Create a buffer to hold the MD5 digest
    unsigned char digest[MD5_DIGEST_LENGTH];

    // Compute the MD5 hash
    MD5((unsigned char*)input.c_str(), input.length(), digest);

    // Convert the digest to a hexadecimal string
    std::ostringstream md5str;
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        md5str << std::hex << std::setw(2) << std::setfill('0') << (int)digest[i];
    }

    return md5str.str();
}

std::string create_hex_string(
    int batch_size,
    int num_inputs,
    int input_a_height,
    int input_a_width,
    const std::string& input_a_dtype,
    const std::string& input_a_layout,
    const std::string& input_a_memory_config,
    const std::string& input_a_sharding_strategy,
    const std::string& multi_core_program_config,
    bool is_scale_causal_mask_hw_dims_softmax,
    bool is_inplace,
    bool is_causal_mask,
    const std::string& input_a_shard_orientation,
    const std::string& input_b_memory_config,
    const std::string& softmax_type
)
{
    std::ostringstream concatenated_string;

    concatenated_string << "batch_sizes(" + std::to_string(batch_size) + ",)";
    concatenated_string << "num_inputs" + std::to_string(num_inputs);
    concatenated_string << "input_a_height" << std::to_string(input_a_height);
    concatenated_string << "input_a_width" << std::to_string(input_a_width);
    concatenated_string << "input_a_dtype" << input_a_dtype;
    concatenated_string << "input_a_layout" << input_a_layout;
    concatenated_string << "input_a_memory_config" << input_a_memory_config;
    concatenated_string << "input_a_sharding_strategy" << input_a_sharding_strategy;
    concatenated_string << "multi_core_program_config" << multi_core_program_config;
    concatenated_string << "is_scale_causal_mask_hw_dims_softmax" << (is_scale_causal_mask_hw_dims_softmax ? "True" : "False");
    concatenated_string << "is_inplace" << (is_inplace ? "True" : "False");
    concatenated_string << "is_causal_mask" << (is_causal_mask ? "True" : "False");
    concatenated_string << "input_a_shard_orientation" << input_a_shard_orientation;
    concatenated_string << "input_b_memory_config" << input_b_memory_config;
    concatenated_string << "softmax_type" << softmax_type;

    return to_hash(concatenated_string.str());
}

std::string create_predefined_hex_string() {
    int batch_size = 1;
    int num_inputs = 1;
    int input_a_height = 1024;
    int input_a_width = 1024;
    std::string input_a_dtype = "DataType.FLOAT32";
    std::string input_a_layout = "Layout.TILE";
    std::string input_a_memory_config = "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt)";
    std::string input_a_sharding_strategy = "ShardStrategy.WIDTH";
    std::string multi_core_program_config = "<class 'ttnn._ttnn.operations.normalization.SoftmaxDefaultProgramConfig'>";
    bool is_scale_causal_mask_hw_dims_softmax = false;
    bool is_inplace = false;
    bool is_causal_mask = false;
    std::string input_a_shard_orientation = "ShardOrientation.COL_MAJOR";
    std::string input_b_memory_config = "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)";
    std::string softmax_type = "softmax";

    return create_hex_string(
        batch_size,
        num_inputs,
        input_a_height,
        input_a_width,
        input_a_dtype,
        input_a_layout,
        input_a_memory_config,
        input_a_sharding_strategy,
        multi_core_program_config,
        is_scale_causal_mask_hw_dims_softmax,
        is_inplace,
        is_causal_mask,
        input_a_shard_orientation,
        input_b_memory_config,
        softmax_type
    );
}

const std::string vector_id1 = "ada59cc88b3dd017c271b40b3fccbb9f72ed34a080f7f92015f259f4";
const std::string vector_id2 = "8c0e346cf8b21f9d36d261d221313f8f33a920961ec887b1ccf99383";
const std::string ELASTIC_DEFAULT_URL = "yyz-elk";
const std::string ELASTIC_PORT = "9200";

std::string get_request(int index)
{
    const std::string ELASTIC_USERNAME = "es_sweeps";
    const std::string ELASTIC_PASSWORD = "RkdH2k*Bhrsd";
    const std::string results_index = "ttnn_sweeps_test_results_softmax";

    const std::string vector_id1 = "ada59cc88b3dd017c271b40b3fccbb9f72ed34a080f7f92015f259f4";
    std::string matches = "[{\"match\": {\"vector_id\": \"" + ((index == 0) ? vector_id1 : vector_id2)  + "\"}}]";

    std::stringstream request;
    request << "GET /" << results_index << "/_search HTTP/1.1\r\n";
    request << "Host: " << ELASTIC_DEFAULT_URL << ":" << ELASTIC_PORT << "\r\n";
    std::string auth = ELASTIC_USERNAME + ":" + ELASTIC_PASSWORD;
    request << "Authorization: Basic " << base64_encode(auth) << "\r\n";
    request << "Content-Type: application/json\r\n";
    std::string body = "{"
                       "\"size\": 10000,"
                       "\"sort\": [{\"timestamp.keyword\": {\"order\": \"asc\"}}],"
                       "\"query\": {\"bool\": {\"must\": " + matches + "}}"
                       "}";

    request << "Content-Length: " << body.size() << "\r\n";
    request << "\r\n";
    request << body;
    return request.str();
}

int main() {
    std::string req0 = get_request(0);
    std::string req1 = get_request(1);
    for (int times=1;times<100;times+=5)
    {
        try {
            std::string response = send_get_request(ELASTIC_DEFAULT_URL, ELASTIC_PORT, req0, req1, times);
            //std::cout << "response:" << response << std::endl;
            std::string status = extract_status(response);
            if (!status.empty()) {
                //std::cout << "Status: " << status << std::endl;
            } else {
                //std::cout << "No status found in response." << std::endl;
            }
        } catch (const std::exception& e) {
            //std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    std::cout << "[";
    for (int i=0;i<durations.size();i++)
    {
        std::cout << durations[i];
        if (i<durations.size()-1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    return 0;
}
