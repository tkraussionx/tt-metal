#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <chrono>
#include <vector>
#include "tt_metal/common/hash_id_maker.hpp"

HashIdMaker hash_id_maker;

struct TestResult {
    std::string sweep_name;
    std::string suite_name;
    std::string vector_id;
    std::string status;
    std::string message;
    std::string e2e_perf;
    std::string timestamp;
    std::string host;
    std::string user;
    std::string git_hash;
};

class JSONParser {
private:
    std::unordered_map<std::string, TestResult> results;

public:
    void parseFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
            return;
        }

        std::string line;
        TestResult currentResult;
        std::string currentKey;
        bool inResultsSection = false;
        bool inObject = false;

        while (std::getline(file, line)) {
            line = trim(line);

            if (line == "\"results\": {") {
                inResultsSection = true;
                continue;
            }

            if (inResultsSection && line.find('{') != std::string::npos) {
                inObject = true;
                continue;
            }

            if (inObject && line.find('}') != std::string::npos) {
                if (!currentResult.vector_id.empty()) {
                    results[currentResult.vector_id] = currentResult;
                }
                inObject = false;
                currentResult = TestResult();
                continue;
            }

            if (inResultsSection && inObject) {
                size_t colonPos = line.find(':');
                if (colonPos == std::string::npos) continue;

                std::string key = trim(line.substr(0, colonPos - 1));
                std::string value = trim(line.substr(colonPos + 1));

                key = removeQuotesAndCommas(key);
                value = removeQuotesAndCommas(value);
                if (key == "vector_id") currentResult.vector_id = value;
                else if (key == "sweep_name") currentResult.sweep_name = value;
                else if (key == "suite_name") currentResult.suite_name = value;
                else if (key == "status") currentResult.status = value;
                else if (key == "message") currentResult.message = value;
                else if (key == "e2e_perf") currentResult.e2e_perf = value;
                else if (key == "timestamp") currentResult.timestamp = value;
                else if (key == "host") currentResult.host = value;
                else if (key == "user") currentResult.user = value;
                else if (key == "git_hash") currentResult.git_hash = value;
            }
        }
        file.close();
    }

    std::string queryTestStatus(const std::string& vector_id) const {
        auto it = results.find(vector_id);
        if (it != results.end()) {
            return it->second.status;
        } else {
            return "Vector ID not found.";
        }
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
        const std::string& buffer_str_2)
    {
        std::string vector_id = hash_id_maker.create_string_for_softmax(
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
        std::string status = queryTestStatus(vector_id);
        return status == "TestStatus.PASS";
    }

private:
    // Helper functions to trim whitespace and remove quotes
    std::string trim(const std::string& str) {
        const std::string whitespace = " \t\n\r";
        size_t start = str.find_first_not_of(whitespace);
        size_t end = str.find_last_not_of(whitespace);
        return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
    }

    std::string removeQuotesAndCommas(const std::string& str) {
        std::string str_without_quotes = "";
        for (int i=0;i<str.length();i++)
        {
            if (str[i]!='\'' && str[i]!='"' && str[i] != ',')
            {
                str_without_quotes+=str[i];
            }
        }
        return str_without_quotes;
    }
};

int main() {
    JSONParser parser;
    parser.parseFile("database.json");
    bool sol =
        parser.does_softmax_pass(
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
    return 0;
}
