// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include <compare>

template <typename T, class Tag>
class TemplateId final {
public:
    explicit TemplateId(T id) noexcept : m_id(id) {}

    // Copy constructor
    TemplateId(const TemplateId& other) noexcept : m_id(other.m_id) {}

    // Copy assignment operator
    TemplateId& operator=(const TemplateId& other) noexcept {
        if (this != &other) {
            m_id = other.m_id;
        }
        return *this;
    }

    // Conversion operator to T
    operator T() const noexcept { return m_id; }

    // Spaceship operator for comparisons
    auto operator<=>(const TemplateId&) const noexcept = default;

private:
    T m_id;
};

class queue_id_tag {};
using QueueId = TemplateId<uint8_t, queue_id_tag>;
const QueueId DefaultQueueId = QueueId(0);
