#include "checks.hpp"
#include "reduction.hpp"
#include "reduction/problem.hpp"
#include "validation.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <cuda/buffer>
#include <limits>
#include <vector>

namespace lg::reduction {
namespace {

bool check_case(Kernel kernel, unsigned int length, unsigned int pattern) {
  const auto problem = make_problem(length, kernel.pad_input ? kernel.segment_size : 1, pattern);
  const auto stream = default_stream();
  auto& pool = device_pool();
  // Early chapter kernels overwrite their input.
  cuda::device_buffer<float> input{stream, pool, problem.input};
  std::vector<float> result{0.0F};
  cuda::device_buffer<float> output{stream, pool, result};
  const auto n = kernel.pad_input ? static_cast<unsigned int>(input.size()) : length;
  if (!LAUNCH_CHECK(kernel.launch(input.data(), output.data(), n, nullptr)) ||
      !copy_checked(output, result)) {
    HOST_LOG(
      "{}: N={} pattern={} padded={} launch/readback failed", kernel.name, length, pattern,
      problem.input.size()
    );
    return false;
  }
  ErrorStats stats;
  stats.add(result[0], problem.expected, 0.0, 0.0);
  if (stats.mismatches == 0) {
    return true;
  }
  HOST_LOG(
    "{}: N={} pattern={} padded={} failed", kernel.name, length, pattern, problem.input.size()
  );
  const std::array expected{problem.expected};
  return verify_values(result, expected, kernel.name, 0.0, 0.0);
}

int check(Kernel kernel) {
  if (kernel.launch == nullptr || kernel.segment_size == 0 ||
      kernel.segment_size > std::numeric_limits<unsigned int>::max() / 5) {
    throw std::invalid_argument("reduction kernel configuration is invalid");
  }
  const auto segment = kernel.segment_size;
  const std::array lengths{0U,
                           1U,
                           2U,
                           3U,
                           31U,
                           32U,
                           33U,
                           255U,
                           256U,
                           257U,
                           511U,
                           512U,
                           513U,
                           segment - 1,
                           segment,
                           segment + 1,
                           2 * segment,
                           2 * segment + 1,
                           5 * segment - 3};
  unsigned int cases = 0;
  unsigned int failed = 0;
  for (const auto length : lengths) {
    if (kernel.max_length != 0 && length > kernel.max_length) {
      continue;
    }
    for (unsigned int pattern = 0; pattern < 5; ++pattern) {
      ++cases;
      try {
        if (!check_case(kernel, length, pattern)) {
          ++failed;
        }
      } catch (const std::exception& error) {
        HOST_LOG("{}: N={} pattern={} failed: {}", kernel.name, length, pattern, error.what());
        ++failed;
      }
    }
  }
  HOST_LOG("{}: {} cases, {} passed, {} failed", kernel.name, cases, cases - failed, failed);
  return failed == 0 ? 0 : 1;
}

} // namespace

int run(int argc, const char* const* argv, Kernel kernel) {
  return guarded([&] {
    if (help_requested(argc, argv)) {
      std::printf("Usage: %s\n", argv[0]);
      return 0;
    }
    if (argc != 1) {
      std::fprintf(stderr, "Unknown reduction argument: %s\n", argv[1]);
      return 1;
    }
    return init_host() ? check(kernel) : 1;
  });
}

} // namespace lg::reduction
