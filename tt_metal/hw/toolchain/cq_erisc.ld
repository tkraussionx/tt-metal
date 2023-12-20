#include "dev_mem_map.h"
#include "cq_erisc_memory.ld"

REGION_ALIAS("REGION_CODE", ERISC_CODE)
REGION_ALIAS("REGION_DATA", ERISC_LOCAL_DATA_MEM)
REGION_ALIAS("REGION_STACK", ERISC_STACK_MEM)

__firmware_stack_size = MEM_ERISC_STACK_SIZE;

#include "tensix-sections.ld"

_Z11kernel_initv = _etext;
