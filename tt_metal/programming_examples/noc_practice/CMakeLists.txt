
set(NOC_EXAMPLES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/unicast_exercise1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unicast_exercise2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unicast_exercise3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multicast_exercise1.cpp
)

CREATE_PGM_EXAMPLES_EXE("${NOC_EXAMPLES_SRCS}" "noc_practice")

add_custom_target(noc_practice DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
