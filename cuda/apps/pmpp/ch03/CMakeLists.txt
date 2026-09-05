FetchContent_Declare(
  lodepng
  URL
    https://github.com/lvandeve/lodepng/archive/ed6fe5825c6a4fbb7f58ab35a4231c7543cd452a.tar.gz
  URL_HASH
    SHA256=c2459a3f9145258f901d262576f7a56ca08087d3b3efeee3ae033c0952120803
)
FetchContent_MakeAvailable(lodepng)

add_library(lodepng STATIC "${lodepng_SOURCE_DIR}/lodepng.cpp")
target_include_directories(lodepng PUBLIC "${lodepng_SOURCE_DIR}")

add_cuda_app(
  KEY pmpp/ch03/grayscale
  SOURCE grayscale.cu
  LIBRARIES lodepng
)
