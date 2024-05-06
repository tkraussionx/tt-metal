#pragma once


namespace tt {
namespace tt_metal {
namespace ccl {

enum Topology {
    Ring = 0,
    Linear = 1,
    Meash = 2
};

struct CCLOpConfig {
    CCLOpConfig(Topology topology) {}

};


class EriscDatamoverBuilder {
   public:
    struct ChannelBufferInterface {
        uint32_t eth_buffer_l1_address;
        uint32_t eth_semaphore_l1_address;
    };

    EriscDatamoverBuilder(AllGatherConfig const& all_gather_config, std::vector<uint32_t> const& local_semaphore_addresses, std::vector<uint32_t> const& local_buffer_addresses, ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode) :
        local_semaphore_addresses(local_semaphore_addresses),
        local_buffer_addresses(local_buffer_addresses),
        eth_buffer_size_bytes(all_gather_config.get_eth_buffer_size()),
        handshake_addr(all_gather_config.get_erisc_handshake_address()),
        num_channel_buffers(all_gather_config.get_num_eth_buffers_per_edm()),
        buffer_sharing_mode(buffer_sharing_mode),
        enable_sender(false),
        enable_receiver(false),
        num_senders(0),
        num_receivers(0)
        {
            active_channels.reserve(num_channel_buffers);
            TT_ASSERT(eth_buffer_size_bytes < 163000);
            log_trace(tt::LogOp, "EriscDatamoverBuilder:");
            for (auto const& addr : local_semaphore_addresses) {
                log_trace(tt::LogOp, "\tsemaphore_address: {}", addr);
            }
            for (auto const& addr : local_buffer_addresses) {
                log_trace(tt::LogOp, "\tbuffer_address: {}", addr);
            }
        }

    [[nodiscard]]
    ChannelBufferInterface add_sender_channel(uint32_t worker_semaphore_address, uint32_t num_eth_messages_to_forward, std::vector<ccl::WorkerXY> const& worker_coords) {
        this->enable_sender = true;
        this->num_senders++;
        auto channel = active_channels.size();
        active_channels.emplace_back(true, worker_semaphore_address, num_eth_messages_to_forward, channel, worker_coords);
        log_trace(tt::LogOp, "Adding sender channel:");
        log_trace(tt::LogOp, "\tworker_semaphore_address: {}", active_channels.back().worker_semaphore_address);
        log_trace(tt::LogOp, "\tnum_eth_messages_to_forward: {}", active_channels.back().num_eth_messages_to_forward);
        log_trace(tt::LogOp, "\tchannel: {}", active_channels.back().channel);
        log_trace(tt::LogOp, "\tis_sender: {}", active_channels.back().is_sender ? 1 : 0);
        log_trace(tt::LogOp, "\tbuffer_address: {}", local_buffer_addresses.at(channel));
        log_trace(tt::LogOp, "\tsemaphore_address: {}", local_semaphore_addresses.at(channel));

        return ChannelBufferInterface{local_buffer_addresses.at(channel), local_semaphore_addresses.at(channel)};
    }
    [[nodiscard]]
    ChannelBufferInterface add_receiver_channel(uint32_t worker_semaphore_address, uint32_t num_eth_messages_to_forward, std::vector<ccl::WorkerXY> const& worker_coords) {
        this->enable_receiver = true;
        this->num_receivers++;
        auto channel = active_channels.size();
        active_channels.emplace_back(false, worker_semaphore_address, num_eth_messages_to_forward, channel, worker_coords);
        log_trace(tt::LogOp, "Adding receiver channel:");
        log_trace(tt::LogOp, "\tworker_semaphore_address: {}", active_channels.back().worker_semaphore_address);
        log_trace(tt::LogOp, "\tnum_eth_messages_to_forward: {}", active_channels.back().num_eth_messages_to_forward);
        log_trace(tt::LogOp, "\tchannel: {}", active_channels.back().channel);
        log_trace(tt::LogOp, "\tis_sender: {}", active_channels.back().is_sender ? 1 : 0);
        return ChannelBufferInterface{local_buffer_addresses.at(channel), local_semaphore_addresses.at(channel)};
    }

    [[nodiscard]]
    std::vector<uint32_t> emit_compile_time_args() const {
        return std::vector<uint32_t>{
            static_cast<uint32_t>(this->enable_sender ? 1 : 0),
            static_cast<uint32_t>(this->enable_receiver ? 1 : 0),
            this->num_senders,
            this->num_receivers,
            this->buffer_sharing_mode};
    }

    [[nodiscard]]
    std::vector<uint32_t> emit_runtime_args() const {
        std::vector<uint32_t> args;
        uint32_t size = 3 + active_channels.size() * 6;
        for (auto const& channel : active_channels) {
            size += channel.worker_coords.size();
        }
        args.reserve(size);

        // Handshake address
        args.push_back(handshake_addr);

        bool senders_below_receivers = active_channels.size() == 0 || this->active_channels.front().is_sender;

        // Sender channel args
        uint32_t sender_channels_offset = senders_below_receivers ? 0 : this->num_receivers;
        args.push_back(sender_channels_offset);
        for (auto const& channel : this->active_channels) {
            if (!channel.is_sender) {
                continue;
            }
            push_back_channel_args(args, channel);
        }

        // Receiver channel args
        uint32_t receiver_channels_offset = senders_below_receivers ? this->num_senders : 0;
        args.push_back(receiver_channels_offset);
        for (auto const& channel : this->active_channels) {
            if (channel.is_sender) {
                continue;
            }
            push_back_channel_args(args, channel);
        }

        return args;
    }

    void dump_to_log() const {
        auto const& rt_args = this->emit_runtime_args();
        log_trace(tt::LogOp, "EDM RT Args:");
        for (auto const& arg : rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
    };

   private:
    struct ChannelBufferSpec {
        ChannelBufferSpec(
            bool is_sender,
            uint32_t worker_semaphore_address,
            uint32_t num_eth_messages_to_forward,
            uint32_t channel,
            std::vector<ccl::WorkerXY> const& worker_coords
        ) :
            worker_coords(worker_coords),
            worker_semaphore_address(worker_semaphore_address),
            num_eth_messages_to_forward(num_eth_messages_to_forward),
            channel(channel),
            is_sender(is_sender) {}

        std::vector<ccl::WorkerXY> const worker_coords;
        uint32_t worker_semaphore_address;
        uint32_t num_eth_messages_to_forward;
        uint32_t channel;
        bool is_sender;
    };

    void push_back_channel_args (std::vector<uint32_t> &args, ChannelBufferSpec const& channel) const {
        args.push_back(this->local_buffer_addresses.at(channel.channel));
        args.push_back(channel.num_eth_messages_to_forward);
        args.push_back(this->eth_buffer_size_bytes);
        args.push_back(this->local_semaphore_addresses.at(channel.channel));
        args.push_back(channel.worker_semaphore_address);
        args.push_back(channel.worker_coords.size());
        for (auto const& worker_coord : channel.worker_coords) {
            args.push_back(worker_coord.to_uint32());
        }
    }

    std::vector<ChannelBufferSpec> active_channels;
    std::vector<uint32_t> const local_semaphore_addresses;
    std::vector<uint32_t> const local_buffer_addresses;
    uint32_t eth_buffer_size_bytes;
    uint32_t handshake_addr;
    uint32_t const num_channel_buffers;
    ccl::EriscDataMoverBufferSharingMode const buffer_sharing_mode;
    uint32_t num_senders;
    uint32_t num_receivers;

    bool enable_sender;
    bool enable_receiver;
};

}; // namespace ccl
}; // namespace tt_metal
}; // namespace tt
