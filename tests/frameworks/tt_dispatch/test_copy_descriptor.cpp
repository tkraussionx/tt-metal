#include "frameworks/tt_dispatch/impl/copy_descriptor.hpp"
#include "tt_metal/host_api.hpp"

bool test_copy_descriptor_prims() {
    CopyDescriptor<100> cd;

    cd.clear();

    bool pass = true;
    cd.add_read(5, 5, 4);


    return pass;
}

int main() {

    bool pass = test_copy_descriptor_prims();
    TT_ASSERT(pass);
}
