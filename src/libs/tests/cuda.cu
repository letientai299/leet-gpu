#include "benchmark/runner.hpp"
#include "checks.hpp"
#include "convolution/benchmark.hpp"
#include "convolution/correctness.hpp"
#include "convolution3d/benchmark.hpp"
#include "convolution3d/correctness.hpp"
#include "matmul/benchmark.hpp"
#include "matmul/correctness.hpp"
#include "stencil/benchmark.hpp"
#include "stencil/correctness.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string_view>

namespace {

void require(bool condition) {
  if (!condition) {
    throw std::runtime_error("CUDA regression failed");
  }
}

__global__ void gemm(const float* a, const float* b, float* c, unsigned m, unsigned n, unsigned k) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m * n) {
    return;
  }
  float value = 0;
  for (unsigned inner = 0; inner < k; ++inner) {
    value += a[index / n * k + inner] * b[inner * n + index % n];
  }
  c[index] = value;
}

void launch_gemm(
  const float* a, const float* b, float* c, const lg::matmul::GemmShape& shape, cudaStream_t stream
) {
  gemm<<<(shape.c_size() + 127) / 128, 128, 0, stream>>>(a, b, c, shape.m(), shape.n(), shape.k());
}

void missing_gemm(const float*, const float*, float*, const lg::matmul::GemmShape&, cudaStream_t) {
}

__global__ void convolution(
  const float* input,
  const float* filter,
  float* output,
  int width,
  int height,
  int depth,
  int radius,
  bool volume
) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= width * height * depth) {
    return;
  }
  const int x = index % width;
  const int y = index / width % height;
  const int z = index / width / height;
  const int side = radius * 2 + 1;
  float value = 0;
  for (int fz = 0; fz < (volume ? side : 1); ++fz) {
    const int iz = volume ? z + fz - radius : 0;
    for (int fy = 0; fy < side; ++fy) {
      const int iy = y + fy - radius;
      for (int fx = 0; fx < side; ++fx) {
        const int ix = x + fx - radius;
        if (ix >= 0 && ix < width && iy >= 0 && iy < height && iz >= 0 && iz < depth) {
          value += input[(iz * height + iy) * width + ix] * filter[(fz * side + fy) * side + fx];
        }
      }
    }
  }
  output[index] = value;
}

void launch_2d(
  const float* input,
  const float* filter,
  float* output,
  lg::convolution::Shape shape,
  cudaStream_t stream
) {
  convolution<<<(shape.input_size() + 127) / 128, 128, 0, stream>>>(
    input, filter, output, shape.width, shape.height, 1, shape.radius, false
  );
}

void missing_2d(const float*, const float*, float*, lg::convolution::Shape, cudaStream_t) {
}

void launch_3d(
  const float* input,
  const float* filter,
  float* output,
  lg::convolution3d::Shape shape,
  cudaStream_t stream
) {
  convolution<<<(shape.input_size() + 127) / 128, 128, 0, stream>>>(
    input, filter, output, shape.width, shape.height, shape.depth, shape.radius, true
  );
}

void missing_3d(const float*, const float*, float*, lg::convolution3d::Shape, cudaStream_t) {
}

__global__ void
stencil(const float* input, float* output, lg::stencil::Shape shape, lg::stencil::Coefficients c) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto plane = shape.plane_size();
  const auto x = static_cast<int>(index % shape.width);
  const auto y = static_cast<int>(index / shape.width % shape.height);
  const auto z = static_cast<int>(index / plane);
  if (index >= shape.grid_size() || x == 0 || y == 0 || z == 0 || x == shape.width - 1 ||
      y == shape.height - 1 || z == shape.depth - 1) {
    return;
  }
  output[index] = c.c0 * input[index] + c.c1 * input[index - 1] + c.c2 * input[index + 1] +
                  c.c3 * input[index - shape.width] + c.c4 * input[index + shape.width] +
                  c.c5 * input[index - plane] + c.c6 * input[index + plane];
}

void launch_stencil(
  const float* input,
  float* output,
  lg::stencil::Shape shape,
  lg::stencil::Coefficients coefficients,
  cudaStream_t stream
) {
  stencil<<<(shape.grid_size() + 127) / 128, 128, 0, stream>>>(input, output, shape, coefficients);
}

void missing_stencil(
  const float*, float*, lg::stencil::Shape, lg::stencil::Coefficients, cudaStream_t
) {
}

void check_device(int device) {
  require(CUDA_CHECK(cudaSetDevice(device)));
  for (const auto shape : {lg::matmul::GemmShape{1, 1, 1}, {17, 9, 13}}) {
    lg::matmul::Problem problem(shape);
    lg::matmul::fill_random(problem);
    require(lg::matmul::check(problem, {"good", launch_gemm}) == 0);
    std::fill(problem.a.begin(), problem.a.end(), 0.0F);
    require(lg::matmul::check(problem, {"missing", missing_gemm}) != 0);
  }
  for (const auto shape :
       {lg::convolution::Shape{1, 1, 0}, {2, 3, 3}, {17, 9, 1}, {17, 9, 2}, {17, 9, 3}}) {
    lg::convolution::Problem problem{shape, {}, {}, {}, {}};
    lg::convolution::fill_problem(problem);
    require(lg::convolution::check(problem, {"good", launch_2d}) == 0);
    std::fill(problem.input.begin(), problem.input.end(), 0.0F);
    require(lg::convolution::check(problem, {"missing", missing_2d}) != 0);
    require(lg::convolution::check(problem, {"good", launch_2d}) == 0);
    require(lg::convolution::check(problem, {"missing", missing_2d}) != 0);
  }
  for (const auto shape : {lg::convolution3d::Shape{1, 1, 1, 0}, {2, 3, 4, 3}, {17, 9, 5, 2}}) {
    lg::convolution3d::Problem problem{shape, {}, {}, {}, {}};
    lg::convolution3d::fill_problem(problem);
    require(lg::convolution3d::check(problem, {"good", launch_3d}) == 0);
    require(lg::convolution3d::check(problem, {"good", launch_3d}, 17) == 0);
    require(lg::convolution3d::check(problem, {"good", launch_3d}, 1) == 0);
    std::fill(problem.input.begin(), problem.input.end(), 0.0F);
    require(lg::convolution3d::check(problem, {"missing", missing_3d}) != 0);
    require(lg::convolution3d::check(problem, {"missing", missing_3d}, 17) != 0);
  }
  for (const auto shape : {lg::stencil::Shape{3, 3, 3}, {17, 9, 5}}) {
    lg::stencil::Problem problem{shape, {}, {}, {}, {}};
    lg::stencil::fill_problem(problem);
    require(lg::stencil::check(problem, {"good", launch_stencil}) == 0);
    require(lg::stencil::check(problem, {"good", launch_stencil}, 7) == 0);
    require(lg::stencil::check(problem, {"missing", missing_stencil}) != 0);
  }
  lg::benchmark::KernelResources resources;
  require(lg::benchmark::inspect_kernel<gemm>(dim3{128}, 0, resources) == cudaSuccess);
  require(resources.active_blocks_per_sm > 0 && resources.active_warps_per_sm > 0);
  require(
    lg::benchmark::inspect_kernel<gemm>(dim3{0}, 0, resources) == cudaErrorInvalidConfiguration
  );
}

void check_timer_status(nvbench::state& state) {
  const auto check = [&state](double step, bool stable) {
    state.get_summaries().clear();
    const auto add = [&state](const char* tag, double value) {
      state.add_summary(tag).set_float64("value", value);
    };
    add("nv/cold/time/gpu/mean", 40e-6);
    add("nv/cold/time/gpu/stdev/relative", 0.02);
    add("nv/cold/time/gpu/iqr/relative", 0.02);
    add("nv/cold/time/gpu/iqr/absolute", 0.8e-6);
    add("lg/cold/timer_step", step);
    state.add_summary("nv/cold/sample_size").set_int64("value", 100);
    lg::benchmark::finish_summaries(state, lg::benchmark::Workload{{}, 1, 4, {4, 0}, {}});
    require(!state.is_skipped());
    for (const auto& summary : state.get_summaries()) {
      if (summary.get_tag() == "lg/cold/status") {
        const auto status = summary.get_string("value");
        require(status.find(stable ? "stable" : "UNSTABLE") == 0);
        return;
      }
    }
    require(false);
  };
  check(1.024e-6, true);
  check(0.512e-6, false);
  check(0.0, false);
}

void check_benchmark_status(nvbench::state& state, std::string_view mode) {
  if (mode == "timer-status") {
    check_timer_status(state);
  } else if (mode == "skip") {
    state.skip("injected failure");
  } else {
    lg::benchmark::finish_summaries(state, lg::benchmark::Workload{{}, 1, 4, {4, 0}, {}});
  }
}

} // namespace

int main(int argc, char** argv) try {
  if (!init_host()) {
    return 1;
  }
  if (argc > 1) {
    const std::string_view mode = argv[1];
    argv[1] = argv[0];
    if (mode == "conv2d-bench") {
      return lg::convolution::run_app(
        argc - 1, argv + 1,
        {"convolution2d", launch_2d, lg::convolution::basic_traffic,
         lg::benchmark::fixed_resources<lg::convolution::Shape, convolution, 128>},
        {257, 193, 3}
      );
    }
    if (mode == "conv3d-bench") {
      return lg::convolution3d::run_app(
        argc - 1, argv + 1,
        {"convolution3d", launch_3d, lg::convolution3d::basic_traffic,
         lg::benchmark::fixed_resources<lg::convolution3d::Shape, convolution, 128>},
        {64, 33, 17, 3}
      );
    }
    if (mode == "stencil-bench") {
      return lg::stencil::run_app(
        argc - 1, argv + 1,
        {"stencil", launch_stencil, lg::stencil::basic_traffic,
         lg::benchmark::fixed_resources<lg::stencil::Shape, stencil, 128>},
        {64, 33, 17}
      );
    }
    if (mode == "bad-shape") {
      return lg::matmul::run_bench_app(argc - 1, argv + 1, {{"missing", missing_gemm}});
    }
    if (mode == "bench") {
      return lg::matmul::run_bench_app(
        argc - 1, argv + 1,
        {{"baseline",
          launch_gemm,
          lg::matmul::cell_traffic,
          {},
          lg::benchmark::fixed_resources<lg::matmul::GemmShape, gemm, 128>},
         {"candidate",
          launch_gemm,
          lg::matmul::cell_traffic,
          {},
          lg::benchmark::fixed_resources<lg::matmul::GemmShape, gemm, 128>}}
      );
    }
    lg::benchmark::add("failure", [mode](nvbench::state& state, nvbench::type_list<>) {
      check_benchmark_status(state, mode);
    });
    if (mode == "retain") {
      float* buffer = nullptr;
      require(CUDA_CHECK(cudaMalloc(&buffer, sizeof(float))));
      const int result = lg::benchmark::run_args(argc - 1, argv + 1);
      require(result != 0);
      require(CUDA_CHECK(cudaMemset(buffer, 0, sizeof(float))));
      require(CUDA_CHECK(cudaFree(buffer)));
      return 0;
    }
    return lg::benchmark::run_args(argc - 1, argv + 1);
  }
  int devices = 0;
  require(CUDA_CHECK(cudaGetDeviceCount(&devices)));
  for (int device = 0; device < devices; ++device) {
    check_device(device);
  }
  std::puts("CUDA regressions passed");
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
