
set(MATMUL_EXERCISE_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_single_core
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multi_core
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multicore_reuse
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multicore_reuse_mcast
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_block_multicore_reuse_mcast
)

CREATE_PGM_EXAMPLES_EXE("${MATMUL_EXERCISE_SRCS}" "matmul_exercise")

add_custom_target(matmul_exercise DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
