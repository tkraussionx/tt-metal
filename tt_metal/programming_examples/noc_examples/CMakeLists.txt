
set(NOC_EXAMPLES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/unicast_example1.cpp
)

CREATE_PGM_EXAMPLES_EXE("${NOC_EXAMPLES_SRCS}" "noc_examples")

add_custom_target(noc_examples DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
