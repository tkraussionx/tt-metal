#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <chrono>
#include <vector>


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

    auto end1 = std::chrono::high_resolution_clock::now();
    std::vector<std::string> statuses;
    for (int i=0;i<1000000;i++)
    {
        std::string vector_id = "2e1b633d169f4a45995b85bb11da8455";
        std::string status = parser.queryTestStatus(vector_id);
        statuses.push_back(status);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto response_time2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end1).count();
    std::cout << "response_time2=" << response_time2 << std::endl;
    return 0;
}
