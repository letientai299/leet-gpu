add_executable(harness_host host.cpp)
target_link_libraries(harness_host PRIVATE lg_matmul)
add_test(NAME harness_host COMMAND harness_host)

add_executable(harness_cuda cuda.cu)
set_target_properties(
  harness_cuda
  PROPERTIES LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(
  harness_cuda
  PRIVATE
    lg_matmul_nvbench
    lg_convolution_nvbench
    lg_convolution3d_nvbench
    lg_stencil_nvbench
)
add_test(NAME harness_cuda COMMAND harness_cuda)
set_tests_properties(harness_cuda PROPERTIES LABELS gpu)
add_test(NAME harness_retain COMMAND harness_cuda retain --quiet)
set_tests_properties(harness_retain PROPERTIES LABELS gpu)
add_test(NAME harness_timer_status COMMAND harness_cuda timer-status --quiet)
set_tests_properties(harness_timer_status PROPERTIES LABELS gpu)

foreach(mode IN ITEMS skip no-samples bad-shape)
  add_test(NAME harness_${mode} COMMAND harness_cuda ${mode} --quiet)
  set_tests_properties(harness_${mode} PROPERTIES LABELS gpu WILL_FAIL TRUE)
endforeach()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(
  NAME harness_scripts
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts_test.py"
)
