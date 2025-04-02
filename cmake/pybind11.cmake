function(download_pybind11)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  set(pybind11_URL  "https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.tar.gz")
  set(pybind11_HASH "SHA256=bf8f242abd1abcd375d516a7067490fb71abd79519a282d22b6e4d19282185a7")

  FetchContent_Declare(pybind11
    URL               ${pybind11_URL}
    URL_HASH          ${pybind11_HASH}
  )

  FetchContent_GetProperties(pybind11)
  if(NOT pybind11_POPULATED)
    message(STATUS "Downloading pybind11")
    FetchContent_Populate(pybind11)
  endif()
  message(STATUS "pybind11 is downloaded to ${pybind11_SOURCE_DIR}")
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_pybind11()
