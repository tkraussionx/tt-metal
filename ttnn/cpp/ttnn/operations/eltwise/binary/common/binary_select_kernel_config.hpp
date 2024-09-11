// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::operations::binary {

enum class ReaderConfig {
    CONTIGUOUS,
    PERMUTE_A_0312,
    PERMUTE_B_0312,
    PERMUTE_AB_0312,
};

enum class ComputeConfig {
    CONTIGUOUS,
    PERMUTE_0312,
};

enum class WriterConfig {
    CONTIGUOUS,
    PERMUTE_0312,
};

struct SelectKernelConfig {
    ReaderConfig reader = ReaderConfig::CONTIGUOUS;
    ComputeConfig compute = ComputeConfig::CONTIGUOUS;
    WriterConfig writer = WriterConfig::CONTIGUOUS;
};

}
