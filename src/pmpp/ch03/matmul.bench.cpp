void release_matmul_benchmark_data();

#define NVBENCH_MAIN_FINALIZE_CUSTOM_PRE() release_matmul_benchmark_data()
#include "matmul.hpp"

#include <charconv>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace {

struct BenchmarkArgs {
  unsigned height = kMatmulHeight;
  unsigned width = kMatmulWidth;
  unsigned k = kMatmulK;
  std::vector<char*> nvbench;
};

std::unique_ptr<Matmul> host_data;
std::map<int, std::unique_ptr<MatmulDeviceData>> device_data;

bool parse_dimension(std::string_view text, unsigned& value) {
  unsigned parsed = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || parsed == 0 ||
      parsed > static_cast<unsigned>(std::numeric_limits<int>::max())) {
    return false;
  }
  value = parsed;
  return true;
}

bool parse_args(int argc, char** argv, BenchmarkArgs& args) {
  args.nvbench.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    if (arg == "--bench") {
      continue;
    }

    unsigned* value = nullptr;
    if (arg == "--height") {
      value = &args.height;
    } else if (arg == "--width") {
      value = &args.width;
    } else if (arg == "--k") {
      value = &args.k;
    }

    if (value != nullptr) {
      ++index;
      if (index == argc || !parse_dimension(argv[index], *value)) {
        return false;
      }
      continue;
    }

    constexpr std::string_view height_prefix = "--height=";
    constexpr std::string_view width_prefix = "--width=";
    constexpr std::string_view k_prefix = "--k=";
    if (arg.rfind(height_prefix, 0) == 0) {
      if (!parse_dimension(arg.substr(height_prefix.size()), args.height)) {
        return false;
      }
    } else if (arg.rfind(width_prefix, 0) == 0) {
      if (!parse_dimension(arg.substr(width_prefix.size()), args.width)) {
        return false;
      }
    } else if (arg.rfind(k_prefix, 0) == 0) {
      if (!parse_dimension(arg.substr(k_prefix.size()), args.k)) {
        return false;
      }
    } else {
      args.nvbench.push_back(argv[index]);
    }
  }
  return true;
}

MatmulDeviceData& get_device_data(nvbench::state& state) {
  const auto& selected_device = state.get_device();
  if (!selected_device.has_value()) {
    throw std::runtime_error("CUDA device is unavailable");
  }
  const int device = selected_device.value().get_id();
  auto found = device_data.find(device);
  if (found == device_data.end()) {
    if (!CUDA_CHECK(cudaSetDevice(device))) {
      throw std::runtime_error("CUDA device selection failed");
    }
    found =
        device_data.emplace(device, std::make_unique<MatmulDeviceData>(*host_data, device)).first;
  }
  return *found->second;
}

void print_usage(const char* app) {
  std::printf("Usage: %s [--height N] [--width N] [--k N] [NVBench options]\n", app);
  std::printf("Defaults: --height %u --width %u --k %u\n", kMatmulHeight, kMatmulWidth, kMatmulK);
}

} // namespace

void matmul_nvbench(nvbench::state& state, MatmulKernel launch) {
  benchmark_matmul(state, get_device_data(state), launch, true);
}

void release_matmul_benchmark_data() {
  for (auto& [device, data] : device_data) {
    CUDA_CHECK(cudaSetDevice(device));
    data.reset();
  }
  device_data.clear();
  host_data.reset();
}

int main(int argc, char** argv) {
  BenchmarkArgs args;
  if (!parse_args(argc, argv, args)) {
    print_usage(argv[0]);
    return 2;
  }
  if (std::find(args.nvbench.begin() + 1, args.nvbench.end(), std::string_view{"--help"}) !=
      args.nvbench.end()) {
    print_usage(argv[0]);
  }
  if (!init_host()) {
    return 1;
  }

  host_data = std::make_unique<Matmul>(args.height, args.width, args.k);
  host_data->fill();
  HOST_LOG("Benchmark shape: height %u, width %u, k %u", args.height, args.width, args.k);
  return run_nvbench_args(static_cast<int>(args.nvbench.size()), args.nvbench.data());
}
