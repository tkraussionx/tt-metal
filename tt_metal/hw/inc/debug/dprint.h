// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
* Device-side debug print API for device kernels.
* Works on either one of NC/BR/TR threads.
* On device the use is as follows:
*
* DPRINT << SETW(2) << 0 << 0.1f << "string" << ENDL();
*
* This DebugPrinter object can be created multiple times.
*
* On the host it's required to start the print server first, otherwise the behavior will be incorrect.
* This is because the host print server writes a special value that is used in DebugPrinter() constructor
* to initialize the read/write pointers to 0 only once.
* It is also needed to empty the print buffer, otherwise device code will stall waiting on the host to flush it.
*
* Use impl/debug/dprint_server.h APIs to start the host-side print server.
*
*/

#include <cstdint>
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC)
// TODO(AP): this ifdef doesn't seem to make sense given we include risc_common.h
// The issue is some files included inside risc_common.h only apply to NC/BRISCS
// But moving this ifdef inside of the header breaks other code
// So there are some not fully decoupled dependencies in this header.
#include "risc_common.h"
#endif
#include "hostdevcommon/dprint_common.h"
#include "hostdevcommon/common_runtime_address_map.h"

#include "dprint_buffer.h"
#if defined(COMPILE_FOR_ERISC)
#include "ethernet/tunneling.h"
#endif

#define DPRINT DebugPrinter()

#ifdef UCK_CHLKC_UNPACK
#define DPRINT_UNPACK(x) x
#else
#define DPRINT_UNPACK(x)
#endif

#ifdef UCK_CHLKC_MATH
#define DPRINT_MATH(x) x
#else
#define DPRINT_MATH(x)
#endif

#ifdef UCK_CHLKC_PACK
#define DPRINT_PACK(x) x
#else
#define DPRINT_PACK(x)
#endif

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#define DPRINT_DATA0(x) if(noc_index == 0) x
#define DPRINT_DATA1(x) if(noc_index == 1) x
#else
#define DPRINT_DATA0(x)
#define DPRINT_DATA1(x)
#endif
namespace internal_ {
void __attribute__((section("code_l1"))) risc_context_switch();
}

struct BF16 { uint16_t val; BF16(uint16_t val) : val(val) {} } ATTR_PACK;
struct F32  { float val; F32(float val) : val(val) {} } ATTR_PACK;
struct U32  { uint32_t val; U32(uint32_t val) : val(val) {} } ATTR_PACK;

struct ENDL { char tmp; } ATTR_PACK; // Analog of cout << std::endl - not making it zero size to avoid special cases
struct SETPRECISION { char p; SETPRECISION(char pa) : p(pa) {} } ATTR_PACK; // Analog of cout << std::setprecision()
struct FIXED { char tmp; } ATTR_PACK; // Analog of cout << std::fixed
struct DEFAULTFLOAT { char tmp; } ATTR_PACK; // Analog of cout << std::defaultfloat
struct HEX  { char tmp; } ATTR_PACK; // Analog of cout << std::hex
struct OCT  { char tmp; } ATTR_PACK; // Analog of cout << std::oct
struct DEC  { char tmp; } ATTR_PACK; // Analog of cout << std::dec
struct SETW { char w; SETW(char w) : w(w) {} } ATTR_PACK; // Analog of cout << std::setw()

// These primitives are intended for ordering debug prints
// A possible use here is to synchronize debug print order between cores/harts
// It could be implemented, for instance as code = linearize({x,y})*5 + hart_id
// With another core/hart waiting on that index
struct RAISE { uint32_t code; RAISE(uint32_t val) : code(val) {} } ATTR_PACK; // raise a condition with specified code
struct WAIT { uint32_t code; WAIT(uint32_t val) : code(val) {} } ATTR_PACK; // wait for a condition with specified code

// didn't want to include string.h
inline uint32_t DebugPrintStrLen(const char* val) {
    const char* end = val;
    while (*end) { end++; };
    return uint32_t(end-val)+1;
}

inline uint32_t DebugPrintStrCopy(char* dst, const char* src) {
    uint32_t len = DebugPrintStrLen(src);
    for (uint32_t j = 0; j < len; j++)
        dst[j] = src[j];
    return len;
}

// Extend with new type id here, each new type needs specializations for 1 (or 3) of these functions below:
// This template instantiation maps from type to type id to send over our comm channel
template<typename T> uint8_t DebugPrintTypeToId();
template<typename T> uint32_t DebugPrintTypeToSize(T val) { return sizeof(T); };
template<typename T> const uint8_t* DebugPrintTypeAddr(T* val) { return reinterpret_cast<const uint8_t*>(val); }

template<> uint8_t DebugPrintTypeToId<const char*>()   { return DPrintCSTR; }
template<> uint8_t DebugPrintTypeToId<char*>()         { return DPrintCSTR; }
template<> uint8_t DebugPrintTypeToId<ENDL>()          { return DPrintENDL; }
template<> uint8_t DebugPrintTypeToId<SETW>()          { return DPrintSETW; }
template<> uint8_t DebugPrintTypeToId<uint8_t>()       { return DPrintUINT8; }
template<> uint8_t DebugPrintTypeToId<uint16_t>()      { return DPrintUINT16; }
template<> uint8_t DebugPrintTypeToId<uint32_t>()      { return DPrintUINT32; }
template<> uint8_t DebugPrintTypeToId<uint64_t>()      { return DPrintUINT64; }
template<> uint8_t DebugPrintTypeToId<int8_t>()        { return DPrintINT8; }
template<> uint8_t DebugPrintTypeToId<int16_t>()       { return DPrintINT16; }
template<> uint8_t DebugPrintTypeToId<int32_t>()       { return DPrintINT32; }
template<> uint8_t DebugPrintTypeToId<int64_t>()       { return DPrintINT64; }
template<> uint8_t DebugPrintTypeToId<int>()           { return DPrintINT32; }
template<> uint8_t DebugPrintTypeToId<float>()         { return DPrintFLOAT32; }
template<> uint8_t DebugPrintTypeToId<char>()          { return DPrintCHAR; }
template<> uint8_t DebugPrintTypeToId<RAISE>()         { return DPrintRAISE; }
template<> uint8_t DebugPrintTypeToId<WAIT>()          { return DPrintWAIT; }
template<> uint8_t DebugPrintTypeToId<BF16>()          { return DPrintBFLOAT16; }
template<> uint8_t DebugPrintTypeToId<SETPRECISION>()  { return DPrintSETPRECISION; }
template<> uint8_t DebugPrintTypeToId<FIXED>()         { return DPrintFIXED; }
template<> uint8_t DebugPrintTypeToId<DEFAULTFLOAT>()  { return DPrintDEFAULTFLOAT; }
template<> uint8_t DebugPrintTypeToId<HEX>()           { return DPrintHEX; }
template<> uint8_t DebugPrintTypeToId<OCT>()           { return DPrintOCT; }
template<> uint8_t DebugPrintTypeToId<DEC>()           { return DPrintDEC; }
template<> uint8_t DebugPrintTypeToId<F32>()           { return DPrintFLOAT32; }
template<> uint8_t DebugPrintTypeToId<U32>()           { return DPrintUINT32; }
static_assert(sizeof(int) == 4);

// Specializations for const char* (string literals), typically you will not need these for other types
template<> uint32_t       DebugPrintTypeToSize<const char*>(const char* val) { return DebugPrintStrLen(val); } // also copy the terminating zero
template<> const uint8_t* DebugPrintTypeAddr<const char*>(const char** val)  { return reinterpret_cast<const uint8_t*>(*val); }
template<> uint32_t       DebugPrintTypeToSize<char*>(char* val)             { return DebugPrintStrLen(val); } // also copy the terminating zero
template<> const uint8_t* DebugPrintTypeAddr<char*>(char** val)              { return reinterpret_cast<const uint8_t*>(*val); }


struct DebugPrinter {
    volatile tt_l1_ptr uint32_t* wpos() {
        auto printbuf = get_debug_print_buffer();
        return &reinterpret_cast<DebugPrintMemLayout*>(printbuf)->aux.wpos;
    }
    volatile tt_l1_ptr uint32_t* rpos() {
        auto printbuf = get_debug_print_buffer();
        return &reinterpret_cast<DebugPrintMemLayout*>(printbuf)->aux.rpos;
    }
    uint8_t* buf() { return get_debug_print_buffer(); }
    uint8_t* data() { return reinterpret_cast<DebugPrintMemLayout*>(buf())->data; }
    uint8_t* bufend() { return buf() + PRINT_BUFFER_SIZE; }

    DebugPrinter() {
#if defined(DEBUG_PRINT_ENABLED)
        if (*wpos() == DEBUG_PRINT_SERVER_STARTING_MAGIC) {
            // Host debug print server writes this value
            // we don't want to reset wpos/rpos to 0 unless this is the first time
            // DebugPrinter() is created (even across multiple kernel calls)
            *wpos() = 0;
            *rpos() = 0;
        }
#endif // ENABLE_DEBUG_PRINT
    }
};

struct DebugPrintData {
    uint32_t sz;
    const uint8_t* data_ptr;
    uint8_t type_id;
};

__attribute__((__noinline__))
void debug_print(DebugPrinter &dp, DebugPrintData data) {
    if (*dp.wpos() == DEBUG_PRINT_SERVER_DISABLED_MAGIC) {
        // skip all prints if this hart+core was not specifically enabled on the host
        return;
    }

    uint32_t payload_sz = data.sz;
    const uint8_t *valaddr = data.data_ptr;
    uint8_t typecode = data.type_id;
    constexpr int code_sz = 1; // size of type code
    constexpr int sz_sz = 1; // size of serialized size
    uint32_t wpos = *dp.wpos(); // copy wpos into local storage
    auto sum_sz = payload_sz + code_sz + sz_sz;
    if (dp.data() + wpos + sum_sz >= dp.bufend()) {
        // buffer is full - wait for the host reader to flush+update rpos
        while (*dp.rpos() < *dp.wpos()) {
#if defined(COMPILE_FOR_ERISC)
            internal_::risc_context_switch();
#endif
            ; // wait for host to catch up to wpos with it's rpos
        }
        *dp.wpos() = 0;
        // TODO(AP): are these writes guaranteed to be ordered?
        *dp.rpos() = 0;
        wpos = 0;
        if (payload_sz >= sizeof(DebugPrintMemLayout::data)-2) {
            // Handle a special case - this value cannot be printed because it cannot fit in the buffer.
            // -2 is for code_sz and sz_sz.
            // Note that the outer if is definitely also true if we got to this inner if.
            // In this case we replace the input value with debug error message.
            // We cannot recursively call operator << from here because it hasn't been defined yet
            // so there's a bit of code duplication here for this special case
            // Another possibility is to wait for the device to flush and print the string piecemeal.
            // As a negative side effect,
            // unfortunately this special case increases the code size generated for each instance of <<.
            uint8_t* printbuf = dp.data();
            payload_sz = DebugPrintStrCopy(
                reinterpret_cast<char*>(printbuf+code_sz+sz_sz),
                debug_print_overflow_error_message);
            printbuf[0] = DPrintCSTR;
            printbuf[code_sz] = payload_sz;
            wpos = payload_sz + sz_sz + code_sz;
            *dp.wpos() = wpos;
            return;
        }
    }

    uint8_t* printbuf = dp.data();
    // no need for a circular buffer since perf is not critical
    printbuf[wpos] = typecode;
    wpos += code_sz;
    printbuf[wpos] = payload_sz;
    wpos += sz_sz;
    for (uint32_t j = 0; j < payload_sz; j++)
        printbuf[wpos+j] = valaddr[j];
    wpos += payload_sz;

    // our message needs to be atomic w.r.t code, size and payload
    // so we only update wpos in the end
    *dp.wpos() = wpos;
}

template<typename T>
__attribute__((__noinline__))
DebugPrinter operator <<(DebugPrinter dp, T val) {
#if defined(DEBUG_PRINT_ENABLED) && !defined(PROFILE_KERNEL)
    DebugPrintData data{
        .sz = DebugPrintTypeToSize<T>(val), // includes terminating 0 for char*
        .data_ptr = DebugPrintTypeAddr<T>(&val),
        .type_id = DebugPrintTypeToId<T>()
    };
    debug_print(dp, data);
#endif // ENABLE_DEBUG_PRINT && !PROFILE_KERNEL
    return dp;
}

// explicit instantiations of operator<<
template DebugPrinter operator<< <const char*>(DebugPrinter dp, const char* val);
template DebugPrinter operator<< <ENDL>(DebugPrinter, ENDL val);
template DebugPrinter operator<< <SETW>(DebugPrinter, SETW val);
template DebugPrinter operator<< <uint8_t>(DebugPrinter, uint8_t val);
template DebugPrinter operator<< <uint16_t>(DebugPrinter, uint16_t val);
template DebugPrinter operator<< <uint32_t>(DebugPrinter, uint32_t val);
template DebugPrinter operator<< <uint64_t>(DebugPrinter, uint64_t val);
template DebugPrinter operator<< <int8_t>(DebugPrinter, int8_t val);
template DebugPrinter operator<< <int16_t>(DebugPrinter, int16_t val);
template DebugPrinter operator<< <int32_t>(DebugPrinter, int32_t val);
template DebugPrinter operator<< <int64_t>(DebugPrinter, int64_t val);
template DebugPrinter operator<< <float>(DebugPrinter, float val);
template DebugPrinter operator<< <char>(DebugPrinter, char val);
template DebugPrinter operator<< <RAISE>(DebugPrinter, RAISE val);
template DebugPrinter operator<< <WAIT>(DebugPrinter, WAIT val);
template DebugPrinter operator<< <FIXED>(DebugPrinter, FIXED val);
template DebugPrinter operator<< <DEFAULTFLOAT>(DebugPrinter, DEFAULTFLOAT val);
template DebugPrinter operator<< <HEX>(DebugPrinter, HEX val);
template DebugPrinter operator<< <OCT>(DebugPrinter, OCT val);
template DebugPrinter operator<< <DEC>(DebugPrinter, DEC val);
template DebugPrinter operator<< <SETPRECISION>(DebugPrinter, SETPRECISION val);
template DebugPrinter operator<< <BF16>(DebugPrinter, BF16 val);
template DebugPrinter operator<< <F32>(DebugPrinter, F32 val);
template DebugPrinter operator<< <U32>(DebugPrinter, U32 val);

#include "dprint_tile.h"
