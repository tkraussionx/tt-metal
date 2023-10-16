
#include "risc_epoch.h"
#include "eth_l1_address_map.h"
#include "tensix.h"
//#include "eth_init.h"
//#include "eth_routing_v2.h"
#include "eth_ss.h"

extern uint32_t __erisc_jump_table;

volatile uint32_t tt_l1_ptr * test_mailbox_ptr = (volatile uint32_t tt_l1_ptr *)(eth_l1_mem::address_map::FIRMWARE_BASE + 0x4);

void (*rtos_context_switch_ptr)();
volatile uint32_t *RtosTable = (volatile uint32_t *) &__erisc_jump_table;    //Rtos Jump Table. Runtime application needs rtos function handles.;

#define NOC_X(x) (loading_noc == 0 ? (x) : (noc_size_x-1-(x)))
#define NOC_Y(y) (loading_noc == 0 ? (y) : (noc_size_y-1-(y)))

volatile uint32_t noc_read_scratch_buf[32] __attribute__((aligned(32))) ;
uint64_t my_q_table_offset;
uint32_t my_q_rd_ptr;
uint32_t my_q_wr_ptr;
uint8_t my_x[NUM_NOCS];
uint8_t my_y[NUM_NOCS];
uint8_t my_logical_x[NUM_NOCS];
uint8_t my_logical_y[NUM_NOCS];
uint8_t loading_noc;
uint8_t noc_size_x;
uint8_t noc_size_y;
uint8_t noc_trans_table_en;

uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];
uint32_t noc_xy_local_addr[NUM_NOCS];


constexpr static uint32_t get_arg_addr(int arg_idx) {
      // args are 4B in size
      return eth_l1_mem::address_map::L1_ARG_BASE + (arg_idx << 2);
  }

template <typename T>
inline T get_arg_val(int arg_idx) {
    // only 4B args are supported (eg int32, uint32)
    static_assert("Error: only 4B args are supported" && sizeof(T) == 4);
    return *((volatile tt_l1_ptr T*)(get_arg_addr(arg_idx)));
}

void __attribute__((section("code_l1"))) risc_context_switch()
{
    ncrisc_noc_full_sync();
    rtos_context_switch_ptr();
    ncrisc_noc_counters_init();
}

void __attribute__((section("erisc_l1_code"))) ApplicationHandler(void)
{
  rtos_context_switch_ptr = (void (*)())RtosTable[0];
  RISC_POST_STATUS((uint32_t) 0);


  ncrisc_noc_init();

  // Reader kernel
  // noc api
  risc_init();
  int32_t src_addr = eth_l1_mem::address_map::DATA_BUFFER_SPACE_BASE;
  int32_t dst_addr = eth_l1_mem::address_map::DATA_BUFFER_SPACE_BASE;

  do {
      uint32_t num_loops = get_arg_val<uint32_t>(0);
      for (uint32_t i=0; i < num_loops; i++) {
        eth_send_packet(0, i + (src_addr  >> 4), i + (dst_addr >> 4), 1);
      }
      while (!ncrisc_noc_nonposted_writes_flushed(loading_noc, NCRISC_WR_DEF_TRID));
      RISC_POST_STATUS(0x10000000 | (num_loops << 12));
      risc_context_switch();
  } while (true);
}
