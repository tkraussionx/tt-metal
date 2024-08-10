
set(PROGRAMMING_EXAMPLES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/ttinfo/ttinfo
)

include(${PROJECT_SOURCE_DIR}/cmake/helper_functions.cmake)

CREATE_PGM_EXAMPLES_EXE("${PROGRAMMING_EXAMPLES_SRCS}" "")  # no subdir, output binaries straight to build/programming_examples

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/matmul_exercise)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/noc_practice)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/debugging_and_profiling)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vietnam_making_npu_ops1)

add_custom_target(programming_examples DEPENDS matmul_exercise noc_practice debugging_and_profiling vietnam_making_npu_ops1 ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
