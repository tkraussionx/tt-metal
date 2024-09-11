#pragma once

enum class pkt_dest_size_choices_t {
    RANDOM=0,
    SAME_START_RNDROBIN_FIX_SIZE=1
};

static inline std::string to_string(pkt_dest_size_choices_t choice) {
    switch (choice) {
        case pkt_dest_size_choices_t::RANDOM: return "RANDOM";
        case pkt_dest_size_choices_t::SAME_START_RNDROBIN_FIX_SIZE: return "RND_ROBIN_FIX";
        default: return "unexpected";
    }
}
