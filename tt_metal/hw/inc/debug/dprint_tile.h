// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/dprint_common.h"
#include "llk_io.h"

// Printing tiles from CBs requires reading CB config from generated files
#if defined(DEBUG_PRINT_ENABLED) && defined(DEBUG_PRINT_ENABLED)
#include "chlkc_unpack_data_format.h"
#include "chlkc_unpack_tile_dims.h"
#include "chlkc_pack_data_format.h"
#include "chlkc_pack_tile_dims.h"
#endif

// Macros for printing circular buffer internals
#define CB_RD_PTR(id) (cb_interface[id].fifo_rd_ptr<<4) // only valid in unpacker thread
#define CB_RD_LIM(id) ((cb_interface[id].fifo_limit_plus_1-1)<<4)
#define CB_RD_SZ(id) (cb_interface[id].fifo_size<<4)

#define CB_WR_PTR(id) (cb_interface[id].fifo_wr_ptr<<4) // only valid in packer thread
#define CB_PAGE_COUNT(id) (cb_interface[id].fifo_num_pages)
#define CB_PAGE_SIZE(id) (cb_interface[id].fifo_page_size << 4)

//
// Slices/samples elements of a tile 'itile' from cb using a given numpy style slice object SliceRange.
// Sampling happens relative to the current CB read or write pointer.
// This means that for printing a tile read from the front of the CB,
// the DPRINT << TSLICE(...) call has to occur after cb_wait_front and before cb_pop_front
// For the case of printing a tile from the back of the CB
// the DPRINT << TSLICE(...) call has to occur after cb_reserve_back and before cb_push_back.
//
// MAXCOUNT is the size of reserved space in the print buffer
// if the total element data_count produced by the slice spec exceeds MAXCOUNT, it will be truncated
//
typedef bool dprint_tslice_ptr_t;
#define TSLICE_RD_PTR true
#define TSLICE_WR_PTR false
typedef bool dprint_tslice_cb_t;
#define TSLICE_INPUT_CB true
#define TSLICE_OUTPUT_SB false

typedef struct {
    uint32_t tile_dim_r;
    uint32_t tile_dim_c;
    uint32_t tile_size;
    uint32_t face_dim_r;
    uint32_t face_dim_c;
    uint32_t num_faces;
    uint32_t cb_ptr;
    uint8_t data_format;
} tile_info_t;

#if defined(DEBUG_PRINT_ENABLED)
inline tile_info_t get_tile_info(
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    uint8_t cb,
    dprint_tslice_cb_t cb_type,
    dprint_tslice_ptr_t ptr_type
#else
    uint8_t cb
#endif
) {
    tile_info_t info = {0};
#if defined(UCK_CHLKC_PACK)
    info.cb_ptr = CB_WR_PTR(cb);  // PACK only has write pointer
    info.data_format = pack_dst_format[cb];
    info.tile_dim_r = pack_tile_r_dim[cb];
    info.tile_dim_c = pack_tile_c_dim[cb];
    info.tile_size = pack_tile_size[cb];
    info.face_dim_r = pack_tile_face_r_dim[cb];
    info.num_faces = pack_tile_num_faces[cb];
#elif defined(UCK_CHLKC_UNPACK)
    info.cb_ptr = CB_RD_PTR(cb);  // UNPACK only has read pointer
    info.data_format = unpack_src_format[cb];
    info.tile_dim_r = unpack_tile_r_dim[cb];
    info.tile_dim_c = unpack_tile_c_dim[cb];
    info.tile_size = unpack_tile_size[cb];
    info.face_dim_r = unpack_tile_face_r_dim[cb];
    info.num_faces = unpack_tile_num_faces[cb];
#elif defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    // For BRISC/NCRISC, user chooses which pointer, and specifies whether the CB is input/output
    info.cb_ptr = (ptr_type == TSLICE_WR_PTR) ? CB_WR_PTR(cb) : CB_RD_PTR(cb);
    info.data_format = (cb_type == TSLICE_INPUT_CB) ? unpack_src_format[cb] : pack_dst_format[cb];
    info.tile_dim_r = (cb_type == TSLICE_INPUT_CB) ? unpack_tile_r_dim[cb] : pack_tile_r_dim[cb];
    info.tile_dim_c = (cb_type == TSLICE_INPUT_CB) ? unpack_tile_c_dim[cb] : pack_tile_c_dim[cb];
    info.tile_size = (cb_type == TSLICE_INPUT_CB) ? unpack_tile_size[cb] : pack_tile_size[cb];
    info.face_dim_r = (cb_type == TSLICE_INPUT_CB) ? unpack_tile_face_r_dim[cb] : pack_tile_face_r_dim[cb];
    info.num_faces = (cb_type == TSLICE_INPUT_CB) ? unpack_tile_num_faces[cb] : pack_tile_num_faces[cb];
#else
    info.cb_ptr = 0;
    info.data_format = static_cast<uint8_t>(DataFormat::Invalid);
    return info;
#endif
    info.face_dim_c = info.tile_dim_r * info.tile_dim_c / info.num_faces / info.face_dim_r;
    return info;
}
#endif // defined(DEBUG_PRINT_ENABLED)

// Specialization of TileSliceHostDev, with device-side implementation
template <int MAX_BYTES=32*2>
struct TileSlice : TileSliceHostDev<MAX_BYTES> {
    static inline int tilize_rm_index(int i) {
        // map from rm-index to tiled index
        int w = (i&31), h = (i>>5); // RM i -> RM hw
        int iface = int(w>=16) + 2*int(h>=16);
        w &= 15; h &= 15;
        return (iface<<8) + (h<<4) + w;
    }

    __attribute__((__noinline__)) TileSlice(
        uint8_t cb,
        int tile_idx,
        const SliceRange& slice_range,
    // For NCRISC and BRISC, CBs could be inputs or outputs, need user to specify so that we know what the DataFormat
    // is. This isn't a problem for PACK/UNPACK because they always treat CBs as input/output. Additionally, NCRISC and
    // BRISC have access to both rd and wr ptr, let user choose w/ arg.
#if defined(COMPILE_FOR_NCRISC)
        dprint_tslice_cb_t cb_type,
        dprint_tslice_ptr_t ptr_type = TSLICE_WR_PTR,
#elif defined(COMPILE_FOR_BRISC)
        dprint_tslice_cb_t cb_type,
        dprint_tslice_ptr_t ptr_type = TSLICE_RD_PTR,
#endif
        bool endl_rows = true,
        bool print_untilized = true) {
#if defined(DEBUG_PRINT_ENABLED)

        // ERISC, IERISC, MATH all don't have access to CBs, don't need fill out any data for those cases. Host will
        // pick up the return code and show the user a warning accordingly.
#if defined(UCK_CHLKC_MATH)
        this->return_code = DPrintErrorMath;
        return;
#endif
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
        this->return_code = DPrintErrorEthernet;
        return;
#endif

        // Fill out slice info
        this->slice_range = slice_range;
        this->cb_id = cb;
        this->endl_rows = endl_rows;
        this->data_count = 0; // Computed as we parse the data
                              // CB pointer and DataFormat depend on RISC
        this->return_code = DPrintOK;

        // DataFormat, rd/wr pointer, and Tile size all depend on RISC + in/out
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
        tile_info_t tile_info = get_tile_info(cb, ptr_type, cb_type);
#else
        tile_info_t tile_info = get_tile_info(cb);
#endif
        this->cb_ptr = tile_info.cb_ptr;
        this->data_format = tile_info.data_format;

        // If the data format is unsupported or corrupted, don't continue
        if (!is_supported_format(static_cast<DataFormat>(this->data_format))) {
            this->return_code = DPrintErrorUnsupportedFormat;
            return; // Unsupported type, return
        }

        // Move the pointer to the tile at index requested by user
        uint32_t bytes_per_datum = dprint_datum_size(static_cast<DataFormat>(this->data_format));
        bool is_bfp_format = is_bfp(static_cast<DataFormat>(this->data_format));
        this->cb_ptr += tile_idx * tile_info.tile_size;

        // Check for unprintable data, and return error as necessary
        if ((tile_idx + 1) * tile_info.tile_size > CB_PAGE_SIZE(this->cb_id)) {
            this->cb_ptr = CB_PAGE_SIZE(this->cb_id); // Save the page size we weren't expecting so host can read.
            this->return_code = DPrintErrorBadTileIdx;
            return;
        }
        if (this->cb_ptr < L1_UNRESERVED_BASE || this->cb_ptr >= MEM_L1_SIZE) {
            this->return_code = DPrintErrorBadPointer;
            return; // bad tile pointer, return
        }

        // Stride through the data in the CB and place in print data buffer
        uint8_t *cb_data = reinterpret_cast<uint8_t *>(this->cb_ptr);
        bool max_count_exceeded = false;
        uint32_t byte_idx = 0;
        for (uint32_t h = slice_range.h0; h < slice_range.h1; h += slice_range.hs) {
            for (uint32_t w = slice_range.w0; w < slice_range.w1; w += slice_range.ws) {
                // Convert w_idx, h_idx to 1D index using num_rows
                uint32_t i = w + h * tile_info.tile_dim_r;
                if (is_bfp_format) {
                    uint32_t data_offset = tile_info.face_dim_r * tile_info.num_faces;
                    // Write 1 byte exponent before each datum. Need to do this since requested stride could put us on
                    // any of the faces.
                    uint32_t row_in_face = h % tile_info.face_dim_r;
                    uint32_t col_in_face = w % tile_info.face_dim_c;
                    uint32_t face_idx_r = h / tile_info.face_dim_r;
                    uint32_t face_idx_c = w / tile_info.face_dim_c;
                    uint32_t num_faces_c = tile_info.tile_dim_c / tile_info.face_dim_c;
                    uint32_t face_idx = face_idx_r * num_faces_c + face_idx_c;
                    uint32_t exponent_idx = face_idx * tile_info.face_dim_r + row_in_face;
                    this->data[byte_idx++] = cb_data[exponent_idx];
                    for (uint32_t offset = 0; offset < bytes_per_datum; offset++) {
                        uint32_t data_idx = face_idx * tile_info.face_dim_r * tile_info.face_dim_c +
                                            row_in_face * tile_info.face_dim_c + col_in_face;
                        this->data[byte_idx++] = cb_data[data_offset + data_idx];
                    }
                    if (byte_idx - 2 >= MAX_BYTES) {
                        max_count_exceeded = true;
                        break;
                    }
                } else {
                    if (print_untilized)
                        i = TileSlice::tilize_rm_index(i);  // tilize the index
                    for (uint32_t offset = 0; offset < bytes_per_datum; offset++) {
                        this->data[byte_idx++] = cb_data[i * bytes_per_datum + offset];
                        // If we've gone over the maximum data points to print, break
                        if (byte_idx - 1 >= MAX_BYTES) {
                            max_count_exceeded = true;
                            break;
                        }
                    }
                }
                if (max_count_exceeded)
                    break;
                this->data_count++;
            }
            if (max_count_exceeded)
                break;
        }
#endif // DEBUG_PRINT_ENABLED
    }
} ATTR_PACK;

using TSLICE = TileSlice<64>;

template<> uint8_t DebugPrintTypeToId<TileSlice<64>>()  { return DPrintTILESLICE; } // TODO(AP): can we use SFINAE here?
template<> uint8_t DebugPrintTypeToId<TileSlice<128>>()  { return DPrintTILESLICE; } // TODO(AP): can we use SFINAE here?

template DebugPrinter operator<< <TSLICE>(DebugPrinter, TSLICE val);
