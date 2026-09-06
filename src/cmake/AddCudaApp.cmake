function(add_cuda_app key)
  cmake_parse_arguments(APP "" "EXTENSION" "LIBRARIES" ${ARGN})

  if(NOT key MATCHES "^[a-z0-9][a-z0-9./-]*[a-z0-9]$")
    message(FATAL_ERROR "Invalid app key: ${key}")
  endif()
  if(APP_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
      "add_cuda_app unexpected args: ${APP_UNPARSED_ARGUMENTS}"
    )
  endif()

  string(REPLACE "/" "_" target_suffix "${key}")
  string(REPLACE "." "_" target_suffix "${target_suffix}")
  set(target "app_${target_suffix}")
  get_filename_component(output_name "${key}" NAME)
  get_filename_component(output_dir "${key}" DIRECTORY)

  if(NOT APP_EXTENSION)
    set(APP_EXTENSION cu)
  endif()

  add_executable(${target} "${PROJECT_SOURCE_DIR}/src/${key}.${APP_EXTENSION}")
  target_link_libraries(
    ${target}
    PRIVATE leet_gpu_runtime CCCL::CCCL ${APP_LIBRARIES}
  )
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(
      ${target}
      PRIVATE -static-libgcc -static-libstdc++ -Wl,--gc-sections
    )
  endif()
  set_target_properties(
    ${target}
    PROPERTIES
      LINKER_LANGUAGE CUDA
      OUTPUT_NAME "${output_name}"
      RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin/${output_dir}"
  )

  if(STRIP_DEPLOY AND CMAKE_STRIP AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_command(
      TARGET ${target}
      POST_BUILD
      COMMAND "${CMAKE_STRIP}" --strip-unneeded "$<TARGET_FILE:${target}>"
      COMMENT "Stripping ${key}"
      VERBATIM
    )
  endif()
endfunction()
