#include "benchmark/app.hpp"
#include "checks.hpp"
#include "stencil/benchmark.hpp"
#include "stencil/correctness.hpp"

#include <cuda/buffer>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <utility>

namespace lg::stencil {
namespace {

using DeviceBuffer = cuda::device_buffer<float>;

// 7 multiplies and 6 adds per interior point.
constexpr std::size_t kFlopsPerPoint = 13;

struct Domain {
  using Shape = stencil::Shape;
  using Kernel = stencil::Kernel;
  using Problem = stencil::Problem;

  static constexpr Shape kCheckShape = kDefaultShape;
  // run_app already compared every value on the small shape.
  static constexpr std::size_t kBenchSamples = 257;

  struct Device {
    Shape shape;
    Coefficients coefficients;
    DeviceBuffer input;
    DeviceBuffer output;

    explicit Device(const Problem& problem)
        : shape(problem.shape), coefficients(problem.coefficients),
          input(default_stream(), device_pool(), problem.input),
          output(default_stream(), device_pool(), initial_output(problem)) {
    }

    [[nodiscard]] static cudaError_t prepare(const Kernel& /*kernel*/) {
      return cudaSuccess;
    }

    void launch(const Kernel& kernel, cudaStream_t stream) {
      kernel.launch(input.data(), output.data(), shape, coefficients, stream);
    }
  };

  static Problem make_problem(Shape shape) {
    Problem problem{.shape = shape};
    fill_problem(problem);
    return problem;
  }

  static int check(Problem& problem, const Kernel& kernel, std::size_t samples) {
    return stencil::check(problem, kernel, samples);
  }

  static std::optional<lg::benchmark::Workload> workload(const Shape& shape, const Kernel& kernel) {
    auto workload = lg::benchmark::float_workload(
      {{"W", shape.width}, {"H", shape.height}, {"D", shape.depth}},
      checked_mul(shape.interior_size(), kFlopsPerPoint), shape.grid_size(), shape.interior_size()
    );
    // Input and output both span the whole grid.
    const auto buffers = checked_mul(shape.grid_size(), 2 * sizeof(float));
    if (!workload || !buffers) {
      return std::nullopt;
    }
    workload->buffer_bytes = *buffers;
    return lg::benchmark::with_no_reuse(std::move(workload), kernel.traffic, shape);
  }

  static std::string label(const Shape& shape) {
    return fmt::format("{}x{}x{}", shape.width, shape.height, shape.depth);
  }
};

} // namespace

int run_app(int argc, char** argv, Kernel kernel, Shape bench_shape) {
  return lg::benchmark::run_app<Domain>(argc, argv, kernel, bench_shape);
}

} // namespace lg::stencil
