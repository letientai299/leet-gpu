function(add_cuda_app)
  cmake_parse_arguments(
    APP
    ""
    "KEY;SOURCE"
    "SOURCES;ARCHITECTURES;LIBRARIES"
    ${ARGN}
  )

  if(NOT APP_KEY OR NOT APP_SOURCE)
    message(FATAL_ERROR "add_cuda_app requires KEY and SOURCE")
  endif()
  if(NOT APP_KEY MATCHES "^[a-z0-9][a-z0-9/-]*[a-z0-9]$")
    message(FATAL_ERROR "Invalid app key: ${APP_KEY}")
  endif()

  string(REPLACE "/" "_" target_suffix "${APP_KEY}")
  set(target "app_${target_suffix}")
  get_filename_component(output_name "${APP_KEY}" NAME)
  get_filename_component(output_dir "${APP_KEY}" DIRECTORY)

  if(APP_ARCHITECTURES)
    set(architectures "${APP_ARCHITECTURES}")
  else()
    set(architectures "${LEET_GPU_CUDA_ARCHITECTURES}")
  endif()

  add_executable(${target} ${APP_SOURCE} ${APP_SOURCES})
  target_link_libraries(${target} PRIVATE leet_gpu_runtime ${APP_LIBRARIES})
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(${target} PRIVATE -static-libgcc -static-libstdc++)
  endif()
  set_target_properties(
    ${target}
    PROPERTIES
      CUDA_ARCHITECTURES "${architectures}"
      OUTPUT_NAME "${output_name}"
      RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin/${output_dir}"
  )

  if(STRIP_DEPLOY AND CMAKE_STRIP)
    add_custom_command(
      TARGET ${target}
      POST_BUILD
      COMMAND "${CMAKE_STRIP}" "$<TARGET_FILE:${target}>"
      COMMENT "Stripping ${APP_KEY}"
      VERBATIM
    )
  endif()
endfunction()
