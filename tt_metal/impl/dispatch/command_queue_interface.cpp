// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_interface.hpp"

SystemMemoryWriter::SystemMemoryWriter() {
    this->cq_write_interface.fifo_wr_ptr = CQ_START >> 4;  // In 16B words
    this->cq_write_interface.fifo_wr_toggle = 0; // This is used for the edge case where we wrap and our read pointer has not yet moved
}
