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

std::string send_get_request(const std::string& hostname, const std::string& port, const std::string& request) {
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
    std::vector<char> response_buffer;

    // Sending the request directly
    send(sockfd, request.c_str(), request.size(), 0);
    char buffer[4096];
    int bytes_received;

    while ((bytes_received = recv(sockfd, buffer, sizeof(buffer), 0)) > 0) {
        response_buffer.insert(response_buffer.end(), buffer, buffer + bytes_received);
        if (response_buffer[response_buffer.size()-5] == '0') break;
    }

    close(sockfd);
    freeaddrinfo(res);

    std::string response_string(response_buffer.begin(), response_buffer.end());
    std::cout << "response=" << response_string << std::endl;
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
    concatenated_string << "sweep_namesoftmax";
    std::cout << "concatenated_string=" << concatenated_string.str() << std::endl;
    return to_hash(concatenated_string.str());
}

/* Expects dataype as, for example: FLOAT32 */
std::string datatype_from_string(const std::string& datatype_str)
{
    return "DataType." + datatype_str;
}

/* Expects layout as, for example: TILE*/
std::string layout_from_string(const std::string& layout_str)
{
    return "Layout." + layout_str;
}

/* Expects memory layout as <INTERLEAVED, SHARDED>, buffer type as <L1, DRAM>, currently shard_spec not supported*/
std::string memory_config_from_strings(const std::string& memory_layout_str, const std::string& buffer_str)
{
    return "MemoryConfig(memory_layout=TensorMemoryLayout::"+ memory_layout_str +",buffer_type=BufferType::"+ buffer_str +",shard_spec=std::nullopt)";
}

/* Expects shard scpec as <HEIGHT, WIDTH> */
std::string sharding_strategy_from_string(const std::string& shard_strategy_str)
{
    return "ShardStrategy." + shard_strategy_str;
}

/* Expects shard orientation as <COL_MAJOR, ROW_MAJOR> */
std::string shard_orientation_from_string(const std::string& shard_orientation_str)
{
    return "ShardOrientation." + shard_orientation_str;
}

std::string create_string_from_input(
    int batch_size,
    int num_inputs,
    int input_a_height,
    int input_a_width,
    const std::string& datatype_str,
    const std::string& layout_str,
    const std::string& memory_layout_str_1,
    const std::string& buffer_str_1,
    const std::string& shard_strategy_str,
    bool is_scale_causal_mask_hw_dims_softmax,
    bool is_inplace,
    bool is_causal_mask,
    const std::string& shard_orientation_str,
    const std::string& memory_layout_str_2,
    const std::string& buffer_str_2)
{
    std::string input_a_dtype = datatype_from_string(datatype_str);
    std::string input_a_layout = layout_from_string(layout_str);
    std::string input_a_memory_config = memory_config_from_strings(memory_layout_str_1, buffer_str_1);
    std::string input_a_sharding_strategy = sharding_strategy_from_string(shard_strategy_str);
    std::string multi_core_program_config = "<class 'ttnn._ttnn.operations.normalization.SoftmaxDefaultProgramConfig'>";
    std::string input_a_shard_orientation = shard_orientation_from_string(shard_orientation_str);
    std::string input_b_memory_config = memory_config_from_strings(memory_layout_str_2, buffer_str_2);
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
        is_inplace,+
        is_causal_mask,
        input_a_shard_orientation,
        input_b_memory_config,
        softmax_type
    );
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

const std::string ELASTIC_DEFAULT_URL = "yyz-elk";
const std::string ELASTIC_PORT = "9200";

std::string get_request(std::string vector_id)
{
    const std::string ELASTIC_USERNAME = "es_sweeps";
    const std::string ELASTIC_PASSWORD = "RkdH2k*Bhrsd";
    const std::string results_index = "ttnn_sweeps_test_results_softmax";

    std::string matches = "[{\"match\": {\"vector_id\": \"" + vector_id  + "\"}}]";
    std::cout << "vector_id:" << vector_id << std::endl;
    std::stringstream request;
    request << "GET /" << results_index << "/_search HTTP/1.1\r\n";
    request << "Host: " << ELASTIC_DEFAULT_URL << ":" << ELASTIC_PORT << "\r\n";
    std::string auth = ELASTIC_USERNAME + ":" + ELASTIC_PASSWORD;
    request << "Authorization: Basic " << base64_encode(auth) << "\r\n";
    request << "Content-Type: application/json\r\n";
    std::string body = "{"
                       "\"size\": 10000,"
                       "\"query\": {\"bool\": {\"must\": " + matches + "}}"
                       "}";

    request << "Content-Length: " << body.size() << "\r\n";
    request << "\r\n";
    request << body;
    std::cout << "request=" << request.str() << std::endl;
    return request.str();
}

bool does_softmax_pass(
    int batch_size,
    int num_inputs,
    int input_a_height,
    int input_a_width,
    const std::string& datatype_str,
    const std::string& layout_str,
    const std::string& memory_layout_str_1,
    const std::string& buffer_str_1,
    const std::string& shard_strategy_str,
    bool is_scale_causal_mask_hw_dims_softmax,
    bool is_inplace,
    bool is_causal_mask,
    const std::string& shard_orientation_str,
    const std::string& memory_layout_str_2,
    const std::string& buffer_str_2
)
{
    std::string vector_id = create_string_from_input(
        batch_size,
        num_inputs,
        input_a_height,
        input_a_width,
        datatype_str,
        layout_str,
        memory_layout_str_1,
        buffer_str_1,
        shard_strategy_str,
        is_scale_causal_mask_hw_dims_softmax,
        is_inplace,
        is_causal_mask,
        shard_orientation_str,
        memory_layout_str_2,
        buffer_str_2
    );
    std::string request = get_request(vector_id);
    try {
        std::string response = send_get_request(ELASTIC_DEFAULT_URL, ELASTIC_PORT, request);
        std::cout << "response:" << response << std::endl;
        std::string status = extract_status(response);
        if (!status.empty()) {
            std::cout << "Status: " << status << std::endl;
            return status == "TestStatus.PASS" ? true : false;
        } else {
            std::cout << "No status found in response." << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    bool sol =
        does_softmax_pass(
            1,
            1,
            1024,
            1024,
            "FLOAT32",
            "TILE",
            "INTERLEAVED",
            "L1",
            "WIDTH",
            false,
            false,
            false,
            "COL_MAJOR",
            "INTERLEAVED",
            "DRAM"
        );
    std::cout << "sol=" << sol << std::endl;
    /*std::string vector_id = create_predefined_hex_string();
    std::string request = get_request(vector_id);
    try {
        std::string response = send_get_request(ELASTIC_DEFAULT_URL, ELASTIC_PORT, request);
        std::cout << "response:" << response << std::endl;
        std::string status = extract_status(response);
        if (!status.empty()) {
            std::cout << "Status: " << status << std::endl;
        } else {
            std::cout << "No status found in response." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }*/
    return 0;
}
