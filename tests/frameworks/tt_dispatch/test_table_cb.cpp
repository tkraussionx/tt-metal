#include "tt_metal/host_api.hpp"

bool test_table_cb() {
    /*
        This test ensures that we can relay data to DRAM using the
        table CB mechanism
    */
    bool pass = true;

    int pci_express_slot = 0;
    tt_metal::Device *device =
        tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);
    tt::tt_metal::InitializeDevice(device);

    // Sets up the table CB interfaces on host and in device L1
    uint32_t num_tables = 1;
    uint32_t table_size_in_bytes = 40;
    tt::tt_metal::InitializeDispatch(device, 1, table_size_in_bytes);



    return pass;
}

int main() {

}
