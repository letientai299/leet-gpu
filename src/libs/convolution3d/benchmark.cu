#include "benchmark/app.hpp"
#include "checks.hpp"
#include "convolution3d/benchmark.hpp"
#include "convolution3d/correctness.hpp"

#include <cuda/buffer>
#include <fmt/format.h>
#include <optional>
#include <string>

namespace lg::convolution3d {
namespace {

using DeviceBuffer = cuda::device_buffer<float>;

struct Domain {
  using Shape = convolution3d::Shape;
  using Kernel = convolution3d::Kernel;
  using Problem = convolution3d::Problem;

  static constexpr Shape kCheckShape = kDefaultShape;
  // The host reference is too slow for every value of the bench shape.
  static constexpr std::size_t kBenchSamples = 257;

  struct Device {
    Shape shape;
    DeviceBuffer input;
    DeviceBuffer filter;
    DeviceBuffer output;

    explicit Device(const Problem& problem)
        : shape(problem.shape), input(default_stream(), device_pool(), problem.input),
          filter(default_stream(), device_pool(), problem.filter),
          output(default_stream(), device_pool(), problem.input.size(), cuda::no_init) {
    }

    [[nodiscard]] cudaError_t prepare(const Kernel& kernel) const {
      return kernel.filter_setup == nullptr ? cudaSuccess
                                            : kernel.filter_setup(filter.data(), shape, nullptr);
    }

    void launch(const Kernel& kernel, cudaStream_t stream) {
      kernel.launch(input.data(), filter.data(), output.data(), shape, stream);
    }
  };

  static Problem make_problem(Shape shape) {
    Problem problem{shape, {}, {}, {}, {}};
    fill_problem(problem);
    return problem;
  }

  static int check(Problem& problem, const Kernel& kernel, std::size_t samples) {
    return convolution3d::check(problem, kernel, samples);
  }

  static std::optional<lg::benchmark::Workload> workload(const Shape& shape, const Kernel& kernel) {
    const auto count = taps(shape);
    return lg::benchmark::with_no_reuse(
      lg::benchmark::float_workload(
        {{"W", shape.width}, {"H", shape.height}, {"D", shape.depth}, {"r", shape.radius}},
        count ? checked_mul(*count, std::size_t{2}) : std::nullopt,
        checked_add(shape.input_size(), shape.filter_size()), shape.input_size()
      ),
      kernel.traffic, shape
    );
  }

  static std::string label(const Shape& shape) {
    return fmt::format("{}x{}x{}, radius {}", shape.width, shape.height, shape.depth, shape.radius);
  }
};

} // namespace

int run_app(int argc, char** argv, Kernel kernel, Shape bench_shape) {
  return lg::benchmark::run_app<Domain>(argc, argv, kernel, bench_shape);
}

} // namespace lg::convolution3d
