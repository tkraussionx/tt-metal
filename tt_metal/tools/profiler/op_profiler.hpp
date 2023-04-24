namespace tt {

namespace tt_metal {

    class OpProfiler {

    private:
        // Have to initialize these statics in source file
        bool profile_ops = false;
        string profile_folder = "tt_metal/tools/profiler/logs/ops/";

        Profiler op_profiler;

        unordered_map <string, uint32_t> call_counters;

    Public:
        void startProfiling(string op_name)
        {
            if (profile_ops)
            {
                op_profiler.markStart(op_name);
            }
        }

        void stopProfiling(tt::tt_metal::Device *device, tt::tt_metal::Program *program, string op_name, vector<string> op_meta_data_vector)
        {
            if (profile_ops)
            {
                op_profiler.markStop(op_name);

                auto it = call_counters.find(op_name);
                if (it != call_counters.end())
                {
                    call_counters.at(op_name) = 1;
                }
                else
                {
                    call_counters.at(op_name) ++;
                }

                string call_count = to_string(call_counters.at(op_name));

                string op_meta_data_string = call_count;
                for (const auto &op_meta_data : op_meta_data_vector)
                {
                    op_meta_data_string += ("-" + op_meta_data);
                }

                op_profiler.setOutputDir(profile_folder + "/" + op_name);
                op_profiler.dumpHostResults(op_meta_data_string);

                tt_metal::SetProfilerDir(profile_folder + "/" + op_name + "/" + call_count);
                tt_metal::DumpDeviceProfileResults(device, program);
                tt_metal::DumpHostProfileResults(device, program);
            }
        }

        void set_profiler_flag(bool do_profile)
        {
            profile_ops = do_profile;
        }

        void set_profiler_location(string profiler_log_location)
        {
            profile_folder = profiler_log_location;
        }

}
}
