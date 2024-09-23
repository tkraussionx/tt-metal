uint32_t xor_shift(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

uint32_t mod_no_mult(uint32_t value, uint32_t N) {
    if (N < 32)
        value &= 0x1f;
    while (value >= N) {
        value -= N;
    }
    return value;
}

uint32_t cheap_random(uint32_t *state, uint32_t N) {
    uint32_t rand_val = xor_shift(state);

    if ((N & (N - 1)) == 0) {  // Check if N is a power of 2
        return rand_val & (N - 1);  // Fast path for powers of 2
    } else {
        return mod_no_mult(rand_val, N);  // Fallback for other values of N
    }
}
