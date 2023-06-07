#pragma once

#include <filesystem>
#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

    struct OpData {
        string name;
        Profiler profiler = Profiler();
        vector<string> metaDataVector = {};

        int opCallCount;
        int globalCallCount;
        int stackSize;

        vector<string> inputs = {};
        vector<string> outputs = {};
        string mathFidelity = "";
        string parlStrategy = "";
        string preferredName = "";

        OpData (string opName, int opCount, int globalCount, int stackSizeArg) :
            name(opName),
            opCallCount(opCount),
            globalCallCount(globalCount),
            stackSize(stackSizeArg)
        {}
    };

    class OpProfiler {

        private:
            const string unknownOpName = "unknown_op";
            OpData unknownOp = OpData (unknownOpName,0,0,0);

            bool profileOps = false;
            string profileFolder = "tt_metal/tools/profiler/logs/ops/";
            stack<OpData> opStack;

            unordered_map <string, uint32_t> callCounters;
            int globalCallCount = 0;

            int get_call_count_increment (string opNameArg)
            {
                auto it = callCounters.find(opNameArg);
                if (it != callCounters.end())
                {
                    callCounters.at(opNameArg) ++;
                }
                else
                {
                    callCounters[opNameArg] = 1;
                }
                globalCallCount ++;
                return callCounters.at(opNameArg);
            }

            int get_call_count (string opNameArg)
            {
                TT_ASSERT (callCounters.find(opNameArg) != callCounters.end(),
                        "Something is wrong, following op never started: " + opNameArg );
                return callCounters.at(opNameArg);
            }

            void setup_profiling_folders (string opName, int callCount, Profiler& opProfiler)
            {
                tt::tt_metal::SetProfilerDir(profileFolder + "/" + opName + "/" + to_string(callCount));
                opProfiler.setOutputDir(profileFolder + "/" + opName);
            }

            void setup_profiling_folders (string opName, int callCount)
            {
                tt::tt_metal::SetProfilerDir(profileFolder + "/" + opName + "/" + to_string(callCount));
            }

            OpData& get_op_data()
            {
                if (profileOps)
                {
                    TT_ASSERT (opStack.size() > 0, "Something is wrong, cannot get op data, op stack is empty");
                    return opStack.top();
                }
                return unknownOp;
            }

            string join_vector(const vector<string>& strs, string delimiter = "-")
            {
                string ret = "";

                for (auto &str : strs)
                {
                    ret += (str + delimiter);
                }
                ret = ret.substr(0,ret.size()-1);

                return ret;
            }

            string shape_to_str(const array<uint32_t, 4> shape)
            {
                return to_string(shape[0]) + "_" +\
                    to_string(shape[1]) + "_" +\
                    to_string(shape[2]) + "_" +\
                    to_string(shape[3]);
            }

            string tensor_to_str(const Tensor& tensor)
            {
                const unordered_map <Layout, string> layout_to_str = {
                    {Layout::ROW_MAJOR, "ROW_MAJOR"},
                    {Layout::TILE, "TILE"},
                    {Layout::CHANNELS_LAST, "CHANNELS_LAST"}
                };

                const unordered_map <DataType, string> dtype_to_str = {
                    {DataType::BFLOAT16, "BFLOAT16"},
                    {DataType::FLOAT32, "FLOAT32"},
                    {DataType::UINT32, "UINT32"},
                    {DataType::BFLOAT8_B, "BFLOAT8_B"}
                };

                vector<string> tensorStrs = {
                    shape_to_str(tensor.shape()),
                    layout_to_str.at(tensor.layout()),
                    dtype_to_str.at(tensor.dtype()),
                    tensor.on_host() ? "ON_HOST" : "ON_DEVICE"
                };

                return join_vector(tensorStrs, "|");
            }

            vector<pair<string,string>> generate_addiotional_data()
            {
                vector<pair<string,string>> additionalFields = {};

                auto& opData = get_op_data();
                additionalFields.push_back({"Global Call Count", to_string(opData.globalCallCount)});
                additionalFields.push_back({"Call Count", to_string(opData.opCallCount)});
                additionalFields.push_back({"Stack Size", to_string(opData.stackSize)});
                additionalFields.push_back({"Inputs", join_vector(opData.inputs)});
                additionalFields.push_back({"Outputs", join_vector(opData.outputs)});
                additionalFields.push_back({"Math Fidelity", opData.mathFidelity});
                additionalFields.push_back({"Parallelization Strategy", opData.parlStrategy});
                additionalFields.push_back({"Preferred Name", opData.preferredName});
                additionalFields.push_back({"Meta Data", join_vector(opData.metaDataVector)});

                return additionalFields;
            }

            void clear_profiler()
            {
                TT_ASSERT (opStack.size() > 0, "Something is wrong, op stack is empty, clear profiler");

                opStack.pop();

                if (opStack.size() > 0)
                {
                    auto callingOpName = get_op_data().name;
                    auto callingOpCallCount = get_call_count(callingOpName);
                    TT_ASSERT(callingOpCallCount == get_op_data().opCallCount,
                            "Something is wrong, op call count from op stack head does not match the expected");

                    setup_profiling_folders (callingOpName, callingOpCallCount);
                }
                else
                {
                    unknownOp = OpData(unknownOpName, unknownOp.opCallCount + 1, globalCallCount, 0);
                    setup_profiling_folders (unknownOpName, unknownOp.opCallCount);
                }
            }
        public:


            void start_profiling(const string opName)
            {
                if (profileOps)
                {
                    auto callCount = get_call_count_increment(opName);
                    OpData opData = OpData(opName, callCount, globalCallCount, opStack.size() + 1);

                    opData.profiler.markStart(opName);

                    setup_profiling_folders (opName, callCount, opData.profiler);

                    opStack.push(opData);
                }
            }


            void stop_profiling(const string opName)
            {
                if (profileOps)
                {
                    auto& opData = get_op_data();
                    TT_ASSERT (opName == opData.name, "Something is wrong, op name mismatch");

                    auto additionalFields = generate_addiotional_data();
                    opData.profiler.markStop(opName, false);
                    opData.profiler.dumpHostResults(additionalFields);
                    clear_profiler();
                }
            }

            bool get_profiler_flag()
            {
                return profileOps;
            }

            void append_input_data (const Tensor& input)
            {
                get_op_data().inputs.push_back(tensor_to_str(input));
            }

            void append_output_data (const Tensor& output)
            {
                get_op_data().outputs.push_back(tensor_to_str(output));
            }

            void set_math_fidelity (string fidelity)
            {
                get_op_data().mathFidelity = fidelity;
            }

            void set_parallelization_strategy (string strategy)
            {
                get_op_data().parlStrategy = strategy;
            }

            void set_preferred_name (string name)
            {
                get_op_data().preferredName = name;
            }

            void append_meta_data(string metaData)
            {
                if (profileOps)
                {
                    TT_ASSERT (opStack.size() > 0, "Something is wrong, cannot append meta data, op stack is empty");
                    string noDashMetaData = "";
                    for (auto &ch : metaData)
                    {
                        if (ch != '-')
                        {
                            noDashMetaData += ch;
                        }
                        else
                        {
                            noDashMetaData += '_';
                        }
                    }
                    get_op_data().metaDataVector.push_back(noDashMetaData);
                }
            }

            void set_profiler_flag(bool doProfile)
            {
                profileOps = doProfile;
            }

            void set_profiler_location(string profilerLogFolder)
            {
                if (profileOps)
                {
                    TT_ASSERT (!(std::filesystem::is_directory(profilerLogFolder)), "Folder " + profilerLogFolder + " exists. Either rename or remove it");
                    profileFolder = profilerLogFolder;
                }
            }
    };

    namespace profiler
    {
        inline OpProfiler operationProfiler;

        static void start_profiling (const string opName)
        {
            operationProfiler.start_profiling(opName);
        }

        static void stop_profiling (const string opName)
        {
            operationProfiler.stop_profiling(opName);
        }

        static bool get_profiler_flag ()
        {
            return operationProfiler.get_profiler_flag();
        }

        static void append_input_data (const Tensor& input)
        {
            operationProfiler.append_input_data(input);
        }

        static void append_output_data (const Tensor& output)
        {
            operationProfiler.append_output_data(output);
        }

        static void append_meta_data (const string metaData)
        {
            operationProfiler.append_meta_data(metaData);
        }

        static void set_preferred_name (const string name)
        {
            operationProfiler.set_preferred_name(name);
        }

        static void set_math_fidelity (const string fidelity)
        { operationProfiler.set_math_fidelity(fidelity);
        }

        static void set_parallelization_strategy (const string parlStrategy)
        {
            operationProfiler.set_parallelization_strategy(parlStrategy);
        }

        static void set_profiler_flag (bool profilerFlag)
        {
            operationProfiler.set_profiler_flag(profilerFlag);
        }

        static void set_profiler_location (const string profilerLocation)
        {
            operationProfiler.set_profiler_location(profilerLocation);
        }
    }

}
}
