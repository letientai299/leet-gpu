FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/refs/tags/v1.16.0.tar.gz
  URL_HASH
    SHA256=8741753e488a78dd0d0024c980e1fb5b5c85888447e309d9cb9d949bdb52aa3e
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
  lodepng
  URL
    https://github.com/lvandeve/lodepng/archive/ed6fe5825c6a4fbb7f58ab35a4231c7543cd452a.tar.gz
  URL_HASH
    SHA256=c2459a3f9145258f901d262576f7a56ca08087d3b3efeee3ae033c0952120803
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(lodepng)

find_package(CUDAToolkit REQUIRED)

add_library(leet_gpu_runtime STATIC log.cpp)
target_include_directories(
  leet_gpu_runtime
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(leet_gpu_runtime PUBLIC spdlog::spdlog)

add_library(lodepng STATIC "${lodepng_SOURCE_DIR}/lodepng.cpp")
target_include_directories(lodepng PUBLIC "${lodepng_SOURCE_DIR}")

add_library(leet_gpu_image STATIC image.cpp image_bytes.cpp)
target_include_directories(leet_gpu_image PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
  leet_gpu_image
  PUBLIC leet_gpu_runtime CUDA::cudart_static
  PRIVATE lodepng
)
