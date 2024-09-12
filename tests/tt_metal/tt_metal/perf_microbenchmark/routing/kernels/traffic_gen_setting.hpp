#pragma once

#define TX_TEST_IDX_TOT_DATA_WORDS PQ_TEST_MISC_INDEX + 1
#define TX_TEST_IDX_NPKT PQ_TEST_MISC_INDEX + 3
#define TX_TEST_IDX_WORDS_FLUSHED PQ_TEST_MISC_INDEX + 5
#define TX_TEST_IDX_FEW_DATA_WORDS_SENT_ITER PQ_TEST_MISC_INDEX + 7
#define TX_TEST_IDX_MANY_DATA_WORDS_SENT_ITER PQ_TEST_MISC_INDEX + 9
#define TX_TEST_IDX_ZERO_DATA_WORDS_SENT_ITER PQ_TEST_MISC_INDEX + 11
// #define TX_TEST_IDX_ PQ_TEST_MISC_INDEX +
// #define TX_TEST_IDX_ PQ_TEST_MISC_INDEX +

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
