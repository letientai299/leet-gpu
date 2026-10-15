#include "benchmark/app.hpp"
#include "benchmark/runner.hpp"
#include "checks.hpp"
#include "histogram.hpp"
#include "histogram/inputs.hpp"

#include <array>
#include <cuda/buffer>
#include <span>
#include <string>

namespace lg::histogram {
namespace {

using Counts = std::array<unsigned int, kBins>;

struct Case {
  const char* name;
  std::string input;
  Counts expected;
};

bool check(const Case& problem, Launch launch) {
  const std::span bytes{problem.input.c_str(), problem.input.size() + 1};
  const auto stream = default_stream();
  auto& pool = device_pool();
  const cuda::device_buffer<char> input{stream, pool, bytes};
  Counts actual{};
  cuda::device_buffer<unsigned int> output{stream, pool, actual};
  if (!LAUNCH_CHECK(launch(
        input.data(), static_cast<unsigned int>(problem.input.size()), output.data(), nullptr
      )) ||
      !copy_checked(output, actual)) {
    return false;
  }
  if (actual != problem.expected) {
    HOST_LOG("{} histogram mismatch", problem.name);
    return false;
  }
  HOST_LOG("{} passed", problem.name);
  return true;
}

void run_benchmark(nvbench::state& state, Launch launch, Resources resources, const Input& corpus) {
  const auto& device = state.get_device();
  if (!device || !check_status(cudaSetDevice(device->get_id()))) {
    state.skip("CUDA device unavailable");
    return;
  }
  if (!lg::benchmark::add_resources(state, resources)) {
    return;
  }
  const Case problem{"gutenberg-english", read_input(corpus), corpus.bins};
  if (!check(problem, launch)) {
    state.skip("Histogram correctness failed");
    return;
  }
  const auto stream = default_stream();
  auto& pool = device_pool();
  const cuda::device_buffer<char> input{stream, pool, problem.input};
  Counts actual{};
  cuda::device_buffer<unsigned int> output{stream, pool, actual};
  if (!check_status(cudaDeviceSynchronize())) {
    state.skip("CUDA setup failed");
    return;
  }
  state.add_element_count(problem.input.size(), "Characters");
  const auto execute = [&](cudaStream_t timed_stream) {
    if (!check_status(cudaMemsetAsync(output.data(), 0, sizeof(Counts), timed_stream))) {
      throw std::runtime_error("Histogram reset failed");
    }
    launch(input.data(), static_cast<unsigned int>(input.size()), output.data(), timed_stream);
    if (!check_status(cudaGetLastError())) {
      throw std::runtime_error("Histogram launch failed");
    }
  };
  const auto length = static_cast<unsigned int>(input.size());
  const lg::benchmark::Workload workload{{{"Bytes", length}},
                                         0,
                                         input.size() + sizeof(Counts),
                                         {input.size(), 2 * sizeof(Counts)},
                                         std::nullopt};
  lg::benchmark::measure<unsigned int>(state, workload, nullptr, length, execute);
  if (!copy_checked(output, actual) || actual != problem.expected) {
    state.skip("Histogram result mismatch");
  }
}

} // namespace

int run(int argc, char** argv, Launch launch, Resources resources) {
  return guarded([&] {
    const auto usage = [&] {
      std::printf("Usage: %s [--bench [NVBench options]]\n", argv[0]);
    };
    lg::benchmark::AppArgs args;
    if (!lg::benchmark::parse_app_args(argc, argv, args)) {
      usage();
      return 2;
    }
    if (args.help) {
      usage();
      return 0;
    }
    if (!init_host()) {
      return 1;
    }
    const std::array cases{
      Case{"empty", "", {}},
      Case{"ignored", "AZ09 !@[`{\n", {}},
      Case{"alphabet", "abcdefghijklmnopqrstuvwxyz", {4, 4, 4, 4, 4, 4, 2}},
      Case{"boundaries", "adehilmpqtuxyz", {2, 2, 2, 2, 2, 2, 2}},
      Case{"mixed", "aAazZz!", {2, 0, 0, 0, 0, 0, 2}},
      Case{
        "multiple-blocks", std::string(600, 'a') + std::string(13, 'z'), {600, 0, 0, 0, 0, 0, 13}},
    };
    for (const auto& problem : cases) {
      if (!check(problem, launch)) {
        return 1;
      }
    }
    if (!args.bench) {
      return 0;
    }
    const auto directory = input_directory();
    HOST_LOG("Histogram inputs: {}", directory.string());
    const auto corpus = prepare_input(directory);
    lg::benchmark::add(
      "histogram/gutenberg-english", [&](nvbench::state& state, nvbench::type_list<>) {
        run_benchmark(state, launch, resources, corpus);
      }
    );
    return lg::benchmark::run_args(static_cast<int>(args.remaining.size()), args.remaining.data());
  });
}

} // namespace lg::histogram
