cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lilcom)

# Remember to also change the line 3 of ./scripts/conda/lilcom/meta.yaml
set(LILCOM_VERSION "1.8.0")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT APPLE)
  set(LILCOM_RPATH_ORIGIN "$ORIGIN")
else()
  set(LILCOM_RPATH_ORIGIN "@loader_path")
endif()

set(CMAKE_INSTALL_RPATH ${LILCOM_RPATH_ORIGIN})
set(CMAKE_BUILD_RPATH ${LILCOM_RPATH_ORIGIN})

set(BUILD_SHARED_LIBS ON)
# See
# https://stackoverflow.com/questions/33062728/cmake-link-shared-library-on-windows
if(BUILD_SHARED_LIBS AND MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()

option(LILCOM_ENABLE_TESTS "Whether to build tests" ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(pybind11)

include_directories(${CMAKE_SOURCE_DIR})

if(WIN32)
  # disable various warnings for MSVC
  # 4244: 'initializing': conversion from 'float' to 'int32_t',
  # 4267: 'argument': conversion from 'size_t' to 'uint32_t', possible loss of data
  set(disabled_warnings
      /wd4244
      /wd4267
  )
  message(STATUS "Disabled warnings: ${disabled_warnings}")
  foreach(w IN LISTS disabled_warnings)
    string(APPEND CMAKE_CXX_FLAGS " ${w} ")
  endforeach()
endif()

pybind11_add_module(lilcom_extension
  lilcom/lilcom_extension.cc
  lilcom/compression.cc
)

if(UNIX AND NOT APPLE)
  # Fix https://github.com/lhotse-speech/lhotse/issues/800
  target_link_libraries(lilcom_extension PUBLIC "-Wl,-rpath,${LILCOM_RPATH_ORIGIN}/../..")
endif()

if(LILCOM_ENABLE_TESTS)
  add_executable(int_stream_test lilcom/int_stream_test.cc)
  add_executable(bit_stream_test lilcom/bit_stream_test.cc)
  if(NOT WIN32)
    target_link_libraries(int_stream_test m) # -lm
    target_link_libraries(bit_stream_test m) # -lm
  endif()

  enable_testing()
  add_subdirectory(test)
endif()
