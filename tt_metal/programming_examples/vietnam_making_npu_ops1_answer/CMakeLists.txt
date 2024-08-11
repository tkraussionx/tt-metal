
set(MAKING_OPS1_EXAMPLES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/single_core/single_core
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_core/multi_core
)

CREATE_PGM_EXAMPLES_EXE("${MAKING_OPS1_EXAMPLES_SRCS}" "vietnam_making_npu_ops1")

add_custom_target(vietnam_making_npu_ops1 DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
