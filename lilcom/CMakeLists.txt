cmake_minimum_required(VERSION 3.13)

project(prediction_math C)


SET(CMAKE_C_FLAGS_DEBUG "-O0 -Wall -ftrapv -g -DFIXED_MATH_TEST=1")
SET(CMAKE_C_FLAGS_RELEASE "-O3 -Wall -ftrapv -DNDEBUG -DFIXED_MATH_TEST=1")

set(SOURCE
    prediction_math.c
    fixed_math.c
)

add_executable(prediction_math ${SOURCE})

target_link_libraries(prediction_math m)
