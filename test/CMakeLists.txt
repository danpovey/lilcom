function(lilcom_add_py_test source)
  get_filename_component(name ${source} NAME_WE)
  set(name "${name}_py")
  message(STATUS "name: ${name}")

  add_test(NAME ${name}
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
  )

  get_filename_component(lilcom_path ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)

  set_property(TEST ${name}
    PROPERTY ENVIRONMENT "PYTHONPATH=${lilcom_path}:$<TARGET_FILE_DIR:lilcom_extension>:$ENV{PYTHONPATH}"
  )
endfunction()

if(LILCOM_ENABLE_TESTS)
  set(test_srcs
    test_header.py
    test_lilcom.py
    test_speed.py
  )

  foreach(source IN LISTS test_srcs)
    lilcom_add_py_test(${source})
  endforeach()
endif()
