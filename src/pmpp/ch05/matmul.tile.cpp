#include "../ch03/matmul.hpp"

#include "log.hpp"
#include "matmul/app.hpp"
#include "matmul/benchmark.hpp"
#include "matmul/correctness.hpp"
#include "matmul/kernel.hpp"

#include <cstdio>
#include <cuda/cmath>
#include <exception>
#include <limits>
#include <memory>
#include <utility>

namespace {

namespace mm = lg::matmul;

// PMPP 4e Fig. 5.9 uses 16.
constexpr unsigned kTileWidth = 16;
constexpr unsigned Size = 4096;
constexpr unsigned kBenchHeight = Size;
constexpr unsigned kBenchWidth = Size;
constexpr unsigned kBenchK = Size;

// PMPP 4e Fig. 5.9, §5.4–5.5.
__global__ void
matmul_tile_kernel(const float* a, const float* b, float* c, unsigned m, unsigned n, unsigned k) {
  __shared__ float aTile[kTileWidth][kTileWidth];
  __shared__ float bTile[kTileWidth][kTileWidth];
  auto gx = blockIdx.x * blockDim.x + threadIdx.x;
  auto gy = blockIdx.y * blockDim.y + threadIdx.y;
  auto tx = threadIdx.x;
  auto ty = threadIdx.y;

  float tmp = 0;
  auto tileCount = cuda::ceil_div(k, kTileWidth);
  for (auto i = 0; i < tileCount; i++) {
    // load data to SMEM, with bound check consideration.
    auto aGx = i * kTileWidth + tx;
    aTile[ty][tx] = gy < m && aGx < k ? a[gy * k + aGx] : 0;
    auto bGy = ty + i * kTileWidth;
    bTile[ty][tx] = bGy < k && gx < n ? b[n * bGy + gx] : 0;

    // wait for other threads in the tile to load their elements.
    __syncthreads();

    for (auto j = 0; j < kTileWidth; j++) {
      tmp += aTile[ty][j] * bTile[j][tx];
    }

    // wait for all threads to complete their computation
    // before starting next loop and modify the shared tiles.
    __syncthreads();
  }

  if (gy < m && gx < n) {
    c[gy * n + gx] = tmp;
  }
}

void launch_matmul_tile(
    const float* a, const float* b, float* c, const mm::GemmShape& shape, cudaStream_t stream) {
  const unsigned height = shape.m();
  const unsigned width = shape.n();
  const unsigned k = shape.k();
  const dim3 block(kTileWidth, kTileWidth);
  const dim3 grid(cuda::ceil_div(width, block.x), cuda::ceil_div(height, block.y));
  matmul_tile_kernel<<<grid, block, 0, stream>>>(a, b, c, height, width, k);
}

bool tile_traffic(const mm::GemmShape& shape, mm::Traffic& traffic) {
  const std::size_t a_reloads = cuda::ceil_div(shape.n(), kTileWidth);
  const std::size_t b_reloads = cuda::ceil_div(shape.m(), kTileWidth);
  if (shape.a_size() > std::numeric_limits<std::size_t>::max() / a_reloads ||
      shape.b_size() > std::numeric_limits<std::size_t>::max() / b_reloads) {
    return false;
  }
  const std::size_t a_reads = shape.a_size() * a_reloads;
  const std::size_t b_reads = shape.b_size() * b_reloads;
  if (a_reads > std::numeric_limits<std::size_t>::max() - b_reads) {
    return false;
  }
  traffic.global_reads = a_reads + b_reads;
  traffic.global_writes = shape.c_size();
  return true;
}

const mm::Kernel kMatmulTile{"matmul.tile", launch_matmul_tile, tile_traffic};

std::unique_ptr<mm::Benchmark> harness;

void apply_bench_shape(mm::AppArgs& args) {
  if (!args.has_height) {
    args.height = kBenchHeight;
  }
  if (!args.has_width) {
    args.width = kBenchWidth;
  }
  if (!args.has_k) {
    args.k = kBenchK;
  }
}

void print_usage(const char* app) {
  std::printf("Usage: %s [--height N] [--width N] [--k N] [--bench [options]]\n", app);
  std::printf("Check defaults: --height %u --width %u --k %u\n", mm::kDefaultHeight,
              mm::kDefaultWidth, mm::kDefaultK);
  std::printf("Bench defaults: --height %u --width %u --k %u\n", kBenchHeight, kBenchWidth,
              kBenchK);
}

} // namespace

void bench_matmul_cell(nvbench::state& state) {
  harness->run_with_shape(state, kMatmulCell);
}

void bench_matmul_tile(nvbench::state& state) {
  harness->run_with_shape(state, kMatmulTile);
}

#define MATMUL_BENCHMARK(name, function)                                                           \
  NVBENCH_BENCH(function)                                                                          \
      .set_name(name)                                                                              \
      .set_min_samples(20)                                                                         \
      .set_cold_warmup_runs(5)                                                                     \
      .set_batch_target_time(1.0)                                                                  \
      .set_throttle_threshold(0.9F)                                                                \
      .set_throttle_recovery_delay(0.1F)

MATMUL_BENCHMARK("matmul.cell", bench_matmul_cell);
MATMUL_BENCHMARK("matmul.tile", bench_matmul_tile);

#undef MATMUL_BENCHMARK

int main(int argc, char** argv) try {
  mm::AppArgs args;
  if (!mm::parse_args(argc, argv, args) || (!args.bench && args.remaining.size() != 1)) {
    print_usage(argv[0]);
    return 2;
  }
  if (args.help) {
    print_usage(argv[0]);
    return 0;
  }

  if (!mm::start()) {
    return 1;
  }

  const mm::GemmShape check_shape =
      args.bench ? mm::GemmShape(mm::kDefaultHeight, mm::kDefaultWidth, mm::kDefaultK)
                 : args.shape();
  mm::Problem check_problem(check_shape);
  mm::fill_random(check_problem);
  const int result = mm::check(check_problem, kMatmulTile);
  if (result != 0 || !args.bench) {
    return result;
  }

  apply_bench_shape(args);
  mm::Problem bench_problem(args.shape());
  mm::fill_random(bench_problem);
  harness = std::make_unique<mm::Benchmark>(std::move(bench_problem));
  HOST_LOG("Benchmark shape: height %u, width %u, k %u", args.height, args.width, args.k);
  const int bench_result =
      mm::run_nvbench_args(static_cast<int>(args.remaining.size()), args.remaining.data());
  harness.reset();
  return bench_result;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
