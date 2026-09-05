FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/refs/tags/v1.16.0.tar.gz
  URL_HASH
    SHA256=8741753e488a78dd0d0024c980e1fb5b5c85888447e309d9cb9d949bdb52aa3e
)
FetchContent_MakeAvailable(spdlog)

add_library(leet_gpu_runtime STATIC log.cpp)
target_include_directories(
  leet_gpu_runtime
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(leet_gpu_runtime PUBLIC spdlog::spdlog)
