#pragma once

#include "benchmark/runner.hpp"
#include "checks.hpp"
#include "log.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lg::benchmark {

/// Keeps one `T` per (device, key) and frees each on its own device.
template <typename T, typename Key = int> class DeviceCache {
public:
  DeviceCache() = default;
  DeviceCache(const DeviceCache&) = delete;
  DeviceCache& operator=(const DeviceCache&) = delete;

  ~DeviceCache() {
    for (auto& entry : entries_) {
      check_status(cudaSetDevice(entry.device));
      entry.value.reset();
    }
  }

  /// Returns the cached value, building it from `args` on first use.
  template <typename... Args> T& get(int device, const Key& key, Args&&... args) {
    const auto found = std::ranges::find_if(entries_, [&](const auto& entry) {
      return entry.device == device && entry.key == key;
    });
    if (found != entries_.end()) {
      return *found->value;
    }
    if (!check_status(cudaSetDevice(device))) {
      throw std::runtime_error("CUDA device selection failed");
    }
    entries_.push_back({device, key, std::make_unique<T>(std::forward<Args>(args)...)});
    return *entries_.back().value;
  }

private:
  struct Entry {
    int device;
    Key key;
    std::unique_ptr<T> value;
  };
  std::vector<Entry> entries_;
};

/// Splits `--bench` and `--help` from arguments NVBench consumes.
struct AppArgs {
  bool bench = false;
  bool help = false;
  std::vector<char*> remaining;
};

/// Leftovers are typos unless `--bench` hands them to NVBench.
inline bool parse_app_args(int argc, char** argv, AppArgs& args) {
  args.remaining.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    if (arg == "--bench") {
      args.bench = true;
    } else if (arg == "--help" || arg == "-h") {
      args.help = true;
    } else {
      args.remaining.push_back(argv[index]);
    }
  }
  return args.bench || args.remaining.size() == 1;
}

/// Rejects a bad domain at the template argument, not inside `measure`.
/// `kCheckShape` is compared in full; `kBenchSamples` values are compared at
/// the bench shape (0 means every value).
template <typename Domain>
concept BenchDomain = requires(
  typename Domain::Problem& problem,
  const typename Domain::Kernel& kernel,
  const typename Domain::Shape& shape,
  typename Domain::Device& device,
  std::size_t samples,
  cudaStream_t stream
) {
  requires std::constructible_from<typename Domain::Device, const typename Domain::Problem&>;
  { problem.shape } -> std::convertible_to<typename Domain::Shape>;
  requires std::copy_constructible<typename Domain::Kernel>;
  requires std::move_constructible<typename Domain::Problem>;
  { kernel.name } -> std::convertible_to<std::string_view>;
  {
    kernel.resources
  } -> std::convertible_to<cudaError_t (*)(const typename Domain::Shape&, KernelResources&)>;
  { device.prepare(kernel) } -> std::same_as<cudaError_t>;
  device.launch(kernel, stream);
  { Domain::kCheckShape } -> std::convertible_to<typename Domain::Shape>;
  { Domain::kBenchSamples } -> std::convertible_to<std::size_t>;
  { Domain::make_problem(shape) } -> std::same_as<typename Domain::Problem>;
  { Domain::check(problem, kernel, samples) } -> std::same_as<int>;
  { Domain::workload(shape, kernel) } -> std::same_as<std::optional<Workload>>;
  { Domain::label(shape) } -> std::convertible_to<std::string>;
};

/// Checks then benchmarks one kernel of a fixed-shape domain.
template <BenchDomain Domain> class Harness {
public:
  using Kernel = typename Domain::Kernel;
  using Problem = typename Domain::Problem;

  explicit Harness(Problem problem) : host_(std::move(problem)) {
  }

  void run(nvbench::state& state, const Kernel& kernel) {
    const auto& selected = state.get_device();
    if (!selected.has_value()) {
      state.skip("CUDA device is unavailable");
      return;
    }
    const int device = selected->get_id();
    if (!check_status(cudaSetDevice(device)) ||
        Domain::check(host_, kernel, Domain::kBenchSamples) != 0) {
      state.skip("benchmark correctness failed");
      return;
    }
    auto& data = devices_.get(device, 0, host_);
    if (!check_status(data.prepare(kernel))) {
      state.skip("CUDA kernel setup failed");
      return;
    }
    measure(
      state, Domain::workload(host_.shape, kernel), kernel.resources, host_.shape,
      [&](cudaStream_t stream) {
        data.launch(kernel, stream);
      }
    );
  }

private:
  Problem host_;
  DeviceCache<typename Domain::Device> devices_;
};

template <BenchDomain Domain>
int run_app(
  int argc, char** argv, const typename Domain::Kernel& kernel, typename Domain::Shape bench_shape
) {
  return guarded([&] {
    const auto label = Domain::label(bench_shape);
    const auto usage = [&] {
      std::printf("Usage: %s [--bench [NVBench options]]\n", argv[0]);
      std::printf("Bench shape: %s\n", label.c_str());
    };
    AppArgs args;
    if (!parse_app_args(argc, argv, args)) {
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

    auto problem = Domain::make_problem(Domain::kCheckShape);
    if (const int result = Domain::check(problem, kernel, 0); result != 0 || !args.bench) {
      return result;
    }

    Harness<Domain> harness(Domain::make_problem(bench_shape));
    add(kernel.name, [&harness, kernel](nvbench::state& state, nvbench::type_list<>) {
      harness.run(state, kernel);
    });
    HOST_LOG("Benchmark shape: {}", label);
    return run_args(static_cast<int>(args.remaining.size()), args.remaining.data());
  });
}

} // namespace lg::benchmark
