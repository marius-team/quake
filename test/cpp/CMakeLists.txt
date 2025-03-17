set(QUAKE_TEST_HOME ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB SRCS *.cpp)

ADD_EXECUTABLE(quake_tests ${SRCS})

TARGET_LINK_LIBRARIES(quake_tests
        ${PROJECT_NAME}
        gtest gtest_main 
        Arrow::arrow_shared
)

# Include directories propagated from dependencies
target_include_directories(quake_tests
    PRIVATE
    ${ARROW_INCLUDE_DIR}     # Arrow headers (if not already handled by Arrow::Arrow)
)

add_test(NAME quake_tests COMMAND quake_tests WORKING_DIRECTORY ${QUAKE_TEST_HOME})