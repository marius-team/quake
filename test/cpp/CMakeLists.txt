set(DB_TEST_HOME ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB SRCS *.cpp)

ADD_EXECUTABLE(tests ${SRCS})

TARGET_LINK_LIBRARIES(tests
        ${PROJECT_NAME}
        gtest gtest_main
)

add_test(NAME tests COMMAND tests WORKING_DIRECTORY ${DB_TEST_HOME})