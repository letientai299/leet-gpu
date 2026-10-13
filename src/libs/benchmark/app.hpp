#pragma once

#include "benchmark/runner.hpp"
#include "checks.hpp"
#include "log.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <memory>
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
      CUDA_CHECK(cudaSetDevice(entry.device));
      entry.value.reset();
    }
  }

  /// Returns the cached value, building it from `args` on first use.
  template <typename... Args> T& get(int device, const Key& key, Args&&... args) {
    const auto found = std::find_if(entries_.begin(), entries_.end(), [&](const auto& entry) {
      return entry.device == device && entry.key == key;
    });
    if (found != entries_.end()) {
      return *found->value;
    }
    if (!CUDA_CHECK(cudaSetDevice(device))) {
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

/// Checks then benchmarks one kernel of a fixed-shape domain.
///
/// `Domain` supplies:
/// - `Shape`, `Problem` (with `shape`), `Kernel` (with `name` and `resources`)
/// - `Device`, built from `(const Problem&)` on the current device, with
///   `cudaError_t prepare(const Kernel&)` and `void launch(const Kernel&, cudaStream_t)`
/// - `kCheckShape`, compared in full, and `kBenchSamples`, compared at the
///   bench shape (0 means every value)
/// - `Problem make_problem(Shape)`, `int check(Problem&, const Kernel&, std::size_t samples)`,
///   `std::optional<Workload> workload(const Shape&, const Kernel&)`,
///   `std::string label(const Shape&)`
template <typename Domain> class Harness {
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
    if (!CUDA_CHECK(cudaSetDevice(device)) ||
        Domain::check(host_, kernel, Domain::kBenchSamples) != 0) {
      state.skip("benchmark correctness failed");
      return;
    }
    auto& data = devices_.get(device, 0, host_);
    if (!CUDA_CHECK(data.prepare(kernel))) {
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

template <typename Domain>
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
    HOST_LOG("Benchmark shape: %s", label.c_str());
    return run_args(static_cast<int>(args.remaining.size()), args.remaining.data());
  });
}

} // namespace lg::benchmark
