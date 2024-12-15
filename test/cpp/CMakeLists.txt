set(QUAKE_TEST_HOME ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB SRCS *.cpp)

ADD_EXECUTABLE(quake_tests ${SRCS})

TARGET_LINK_LIBRARIES(quake_tests
        ${PROJECT_NAME}
        gtest gtest_main
)

add_test(NAME quake_tests COMMAND quake_tests WORKING_DIRECTORY ${QUAKE_TEST_HOME})