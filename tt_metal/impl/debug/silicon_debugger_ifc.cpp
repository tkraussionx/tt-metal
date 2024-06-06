// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/jit_build/settings.hpp"

namespace tt::tt_metal {

// Create a mutex silicon_debugger_ifc_mutex
std::mutex silicon_debugger_ifc_mutex;
string SILICON_DEBUGER_RUNTIME_DATA_FILE("generated/silicon_debugger/runtime_data.yaml");

FILE *OpenSiliconDebuggerInterfaceFile() {
    FILE *f;
    if ((f = fopen((tt::llrt::OptionsG.get_root_dir() + SILICON_DEBUGER_RUNTIME_DATA_FILE).c_str(), "a")) == nullptr) {
        TT_THROW("Failed to open silicon debugger data file: {}", SILICON_DEBUGER_RUNTIME_DATA_FILE);
    }
    return f;
}

// Initilize file used by the silicon debugger (gdb server)
void InitSiliconDebuggerInterfaceFile() {
    static bool created = false;
    if (created) {
        return;
    }
    created = true;

    FILE *f;
    string fname = tt::llrt::OptionsG.get_root_dir() + SILICON_DEBUGER_RUNTIME_DATA_FILE;
    string parent_dir = std::filesystem::path(fname).parent_path().string();
    std::filesystem::create_directories(parent_dir);

    if ((f = fopen(fname.c_str(), "w")) == nullptr) {
        TT_THROW("Failed to create silicon debugger data file: {}", SILICON_DEBUGER_RUNTIME_DATA_FILE);
    }

    // We log the locations of the mailboxes and the sizes of the launch message fields to be able
    // to extract the information from the memory on the device.
    fprintf(f, "addresses:\n");
    fprintf(f, "  MEM_MAILBOX_BASE: %d\n", MEM_MAILBOX_BASE);
    fprintf(f, "  ERISC_MEM_MAILBOX_BASE: %d\n", eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);
    fprintf(f, "  MEM_IERISC_MAILBOX_BASE: %d\n", MEM_IERISC_MAILBOX_BASE);

    fprintf(f, "variables:\n");
    fprintf(f, "  mailboxes_t.launch.brisc_watcher_kernel_id:\n    offset: %ld\n    size: %ld\n",
        offsetof(mailboxes_t, launch.brisc_watcher_kernel_id), sizeof(launch_msg_t::brisc_watcher_kernel_id));
    fprintf(f, "  mailboxes_t.launch.ncrisc_watcher_kernel_id:\n    offset: %ld\n    size: %ld\n",
        offsetof(mailboxes_t, launch.ncrisc_watcher_kernel_id), sizeof(launch_msg_t::ncrisc_watcher_kernel_id));
    fprintf(f, "  mailboxes_t.launch.triscs_watcher_kernel_id:\n    offset: %ld\n    size: %ld\n",
        offsetof(mailboxes_t, launch.triscs_watcher_kernel_id), sizeof(launch_msg_t::triscs_watcher_kernel_id));

    fprintf(f, "kernels:\n");
    fclose(f);
}

// We log the kernel ID and the paths to the source file and the build output directory
void SiliconDebuggerInterfaceLogKernel(shared_ptr<Kernel> kernel, const tt::tt_metal::JitBuildOptions &build_options) {
    std::lock_guard<std::mutex> lock(silicon_debugger_ifc_mutex); // Allow for concurrent threads to log kernel information
    FILE *gdb_ifc_file = tt::tt_metal::OpenSiliconDebuggerInterfaceFile();
    fprintf(gdb_ifc_file, "  %d:\n    src: %s\n    out: %s\n", kernel->get_watcher_kernel_id(), kernel->kernel_path_file_name().c_str(), build_options.path.c_str());
    fflush(gdb_ifc_file);
    fclose(gdb_ifc_file);
}


}
